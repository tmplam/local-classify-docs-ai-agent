from playwright.async_api import async_playwright
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
import requests
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from typing import List, Dict, Any
import json
import time
import re
from datetime import datetime, timedelta
import threading

load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
serper = GoogleSerperAPIWrapper()

async def playwright_tools():
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright


def push(text: str):
    """Send a push notification to the user"""
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"


def get_file_tools():
    toolkit = FileManagementToolkit(root_dir="data")
    return toolkit.get_tools()


# ========== RATE LIMITER CLASS ==========

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = datetime.now()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < timedelta(seconds=self.time_window)]
            
            # If we're at the limit, wait
            if len(self.requests) >= self.max_requests:
                oldest_request = min(self.requests)
                wait_time = (oldest_request + timedelta(seconds=self.time_window) - current_time).total_seconds()
                if wait_time > 0:
                    print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Remove the oldest request after waiting
                    self.requests = [req_time for req_time in self.requests 
                                   if current_time - req_time < timedelta(seconds=self.time_window)]
            
            # Add current request
            self.requests.append(current_time)

# ========== ACADEMIC RESEARCH TOOLS ==========

class AcademicSearchTools:
    """Academic search tools for citation analysis using Semantic Scholar API with rate limiting"""
    
    def __init__(self):
        self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1"
        # Conservative rate limiting: 10 requests per minute (600 per hour)
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)
        # Additional delay between requests
        self.min_delay = 1.0  # Minimum 1 second between requests
        self.last_request_time = 0
        
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a rate-limited request to the API"""
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Additional delay to be extra safe
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_delay:
                sleep_time = self.min_delay - time_since_last
                print(f"Additional delay: {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
            
            print(f"Making API request to: {url}")
            response = requests.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            
            # Handle rate limit responses
            if response.status_code == 429:
                print("Rate limit hit (429). Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(url, params)  # Retry
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        
    def search_paper_by_title(self, title: str) -> Dict[str, Any]:
        """Search for a paper by title using Semantic Scholar API"""
        try:
            clean_title = title.strip().replace('"', '')
            
            url = f"{self.semantic_scholar_api}/paper/search"
            params = {
                'query': clean_title,
                'fields': 'paperId,title,authors,year,citationCount,referenceCount,url,abstract',
                'limit': 10
            }
            
            data = self._make_request(url, params)
            
            if not data or not data.get('data'):
                return None
            
            # Find exact or close match
            for paper in data['data']:
                if paper['title'].lower() == clean_title.lower():
                    return paper
            
            # If no exact match, find best match by similarity
            best_match = None
            best_score = 0
            for paper in data['data']:
                # Simple similarity check
                title_words = set(clean_title.lower().split())
                paper_words = set(paper['title'].lower().split())
                common_words = title_words.intersection(paper_words)
                score = len(common_words) / len(title_words.union(paper_words))
                
                if score > best_score:
                    best_score = score
                    best_match = paper
            
            if best_score > 0.6:  # At least 60% similarity
                return best_match
            
            return data['data'][0]  # Return first result if no good match
            
        except Exception as e:
            print(f"Error searching paper: {e}")
            return None
    
    def get_citing_papers(self, paper_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Get papers that cite a specific paper with pagination and rate limiting"""
        try:
            citing_papers = []
            offset = 0
            batch_size = 100  # Process in smaller batches
            
            while len(citing_papers) < limit:
                remaining = limit - len(citing_papers)
                current_batch_size = min(batch_size, remaining)
                
                url = f"{self.semantic_scholar_api}/paper/{paper_id}/citations"
                params = {
                    'fields': 'paperId,title,authors,year,citationCount,url,abstract,venue',
                    'limit': current_batch_size,
                    'offset': offset
                }
                
                data = self._make_request(url, params)
                
                if not data or not data.get('data'):
                    break
                
                batch_papers = []
                for citation in data['data']:
                    citing_paper = citation.get('citingPaper', {})
                    if citing_paper and citing_paper.get('title'):
                        batch_papers.append(citing_paper)
                
                if not batch_papers:
                    break
                
                citing_papers.extend(batch_papers)
                offset += current_batch_size
                
                # If we got fewer papers than requested, we've reached the end
                if len(data['data']) < current_batch_size:
                    break
                
                print(f"Retrieved {len(citing_papers)} citing papers so far...")
            
            return citing_papers
            
        except Exception as e:
            print(f"Error getting citing papers: {e}")
            return []

# Create instance of academic tools
academic_search = AcademicSearchTools()

def analyze_citations_by_year(paper_title: str) -> str:
    """Analyze citations for a paper and sort by year (main function for the specific requirement)"""
    try:
        print(f"Searching for paper: {paper_title}")
        
        # Step 1: Find the target paper
        target_paper = academic_search.search_paper_by_title(paper_title)
        
        if not target_paper:
            return f"❌ Could not find paper with title: '{paper_title}'\n\nPlease check the title spelling or try a shorter version of the title."
        
        paper_id = target_paper['paperId']
        
        print(f"Found target paper: {target_paper['title']} (ID: {paper_id})")
        
        # Step 2: Get citing papers with rate limiting
        print("Retrieving citing papers (this may take a while due to rate limiting)...")
        citing_papers = academic_search.get_citing_papers(paper_id, limit=200)  # Reduced limit
        
        if not citing_papers:
            return f"❌ No citing papers found for: '{target_paper['title']}'\n\nThis could mean:\n- The paper is very new\n- It hasn't been cited yet\n- The citation data is not available in Semantic Scholar"
        
        # Step 3: Filter papers with valid years and sort
        valid_papers = []
        for paper in citing_papers:
            if paper.get('year') and paper.get('title') and paper['year'] >= 1990:  # Filter reasonable years
                valid_papers.append(paper)
        
        if not valid_papers:
            return f"❌ No citing papers with valid publication years found for: '{target_paper['title']}'"
        
        # Sort by year (descending - newest first)
        valid_papers.sort(key=lambda x: x.get('year', 0), reverse=True)
        
        # Step 4: Format comprehensive results
        result = f"📚 CITATION ANALYSIS REPORT\n"
        result += f"{'='*60}\n\n"
        
        # Target paper info
        result += f"🎯 TARGET PAPER:\n"
        result += f"Title: {target_paper['title']}\n"
        result += f"Year: {target_paper.get('year', 'N/A')}\n"
        if target_paper.get('authors'):
            authors = ', '.join([author.get('name', 'Unknown') for author in target_paper['authors'][:3]])
            if len(target_paper['authors']) > 3:
                authors += f" et al. ({len(target_paper['authors'])} authors total)"
            result += f"Authors: {authors}\n"
        result += f"Total Citations: {target_paper.get('citationCount', 0)}\n"
        if target_paper.get('url'):
            result += f"URL: {target_paper['url']}\n"
        
        result += f"\n📊 CITATION SUMMARY:\n"
        result += f"Total citing papers found: {len(valid_papers)}\n"
        
        # Group by year for summary
        year_groups = {}
        for paper in valid_papers:
            year = paper.get('year')
            if year not in year_groups:
                year_groups[year] = []
            year_groups[year].append(paper)
        
        result += f"Publication years range: {min(year_groups.keys())} - {max(year_groups.keys())}\n"
        result += f"Years with citations: {len(year_groups)}\n\n"
        
        # Show year distribution
        result += f"📈 CITATIONS BY YEAR (Descending Order):\n"
        result += f"{'='*60}\n\n"
        
        for year in sorted(year_groups.keys(), reverse=True):
            papers_in_year = year_groups[year]
            result += f"🗓️  {year} ({len(papers_in_year)} {'paper' if len(papers_in_year) == 1 else 'papers'})\n"
            result += f"{'-'*40}\n"
            
            # Sort papers within year by citation count
            papers_in_year.sort(key=lambda x: x.get('citationCount', 0), reverse=True)
            
            for i, paper in enumerate(papers_in_year[:10], 1):  # Show max 10 per year to reduce output
                authors_list = paper.get('authors', [])
                if authors_list:
                    first_author = authors_list[0].get('name', 'Unknown')
                    if len(authors_list) > 1:
                        authors_display = f"{first_author} et al."
                    else:
                        authors_display = first_author
                else:
                    authors_display = "Unknown authors"
                
                venue = paper.get('venue', 'Unknown venue')
                citations = paper.get('citationCount', 0)
                
                result += f"   {i:2d}. {paper['title'][:80]}{'...' if len(paper['title']) > 80 else ''}\n"
                result += f"       👥 {authors_display}\n"
                result += f"       📖 {venue} | 📊 {citations} citations\n"
                
                if paper.get('url'):
                    result += f"       🔗 {paper['url']}\n"
                result += f"\n"
            
            if len(papers_in_year) > 10:
                result += f"       ⋮ ... and {len(papers_in_year) - 10} more papers in {year}\n\n"
            
            result += f"\n"
        
        # Add statistics summary
        total_citations_of_citing_papers = sum(paper.get('citationCount', 0) for paper in valid_papers)
        avg_citations = total_citations_of_citing_papers / len(valid_papers) if valid_papers else 0
        
        result += f"📊 STATISTICS:\n"
        result += f"{'='*30}\n"
        result += f"• Most productive year: {max(year_groups.keys(), key=lambda y: len(year_groups[y]))} ({len(year_groups[max(year_groups.keys(), key=lambda y: len(year_groups[y]))])} papers)\n"
        result += f"• Average citations per citing paper: {avg_citations:.1f}\n"
        
        # Find most cited citing paper
        most_cited = max(valid_papers, key=lambda x: x.get('citationCount', 0))
        result += f"• Most cited citing paper: '{most_cited['title'][:50]}...' ({most_cited.get('citationCount', 0)} citations)\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing citations: {str(e)}\n\nPlease try again or check if the paper title is correct."

def search_academic_paper(paper_title: str) -> str:
    """Get detailed information about a specific paper"""
    try:
        paper = academic_search.search_paper_by_title(paper_title)
        if not paper:
            return f"❌ Could not find paper: '{paper_title}'"
        
        result = f"📄 PAPER DETAILS\n"
        result += f"{'='*40}\n\n"
        result += f"📖 Title: {paper['title']}\n"
        result += f"📅 Year: {paper.get('year', 'N/A')}\n"
        
        if paper.get('authors'):
            authors = ', '.join([author.get('name', 'Unknown') for author in paper.get('authors', [])])
            result += f"👥 Authors: {authors}\n"
        
        result += f"📊 Citations: {paper.get('citationCount', 0)}\n"
        result += f"📚 References: {paper.get('referenceCount', 0)}\n"
        result += f"🆔 Paper ID: {paper.get('paperId', 'N/A')}\n"
        
        if paper.get('venue'):
            result += f"📍 Venue: {paper.get('venue')}\n"
        
        if paper.get('abstract'):
            abstract = paper['abstract']
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            result += f"\n📝 Abstract:\n{abstract}\n"
        
        if paper.get('url'):
            result += f"\n🔗 URL: {paper['url']}\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error getting paper details: {str(e)}"

def search_papers_by_author(author_name: str) -> str:
    """Search for papers by a specific author"""
    try:
        url = f"{academic_search.semantic_scholar_api}/author/search"
        params = {
            'query': author_name,
            'fields': 'authorId,name,paperCount,citationCount,papers.title,papers.year,papers.citationCount',
            'limit': 5
        }
        
        data = academic_search._make_request(url, params)
        
        if not data or not data.get('data'):
            return f"❌ No authors found with name: '{author_name}'"
        
        author = data['data'][0]  # Get first match
        
        result = f"👤 AUTHOR: {author.get('name', 'Unknown')}\n"
        result += f"{'='*40}\n"
        result += f"📊 Paper Count: {author.get('paperCount', 0)}\n"
        result += f"📈 Total Citations: {author.get('citationCount', 0)}\n\n"
        
        papers = author.get('papers', [])
        if papers:
            result += f"📚 RECENT PAPERS:\n"
            result += f"{'-'*30}\n"
            
            # Sort by year and citation count
            papers.sort(key=lambda x: (x.get('year', 0), x.get('citationCount', 0)), reverse=True)
            
            for i, paper in enumerate(papers[:10], 1):
                result += f"{i:2d}. {paper.get('title', 'Unknown title')}\n"
                result += f"    📅 {paper.get('year', 'N/A')} | 📊 {paper.get('citationCount', 0)} citations\n\n"
        
        return result
        
    except Exception as e:
        return f"❌ Error searching author: {str(e)}"

# ========== END ACADEMIC TOOLS ==========

async def other_tools():
    push_tool = Tool(name="send_push_notification", func=push, description="Use this tool when you want to send a push notification")
    file_tools = get_file_tools()

    tool_search = Tool(
        name="search",
        func=serper.run,
        description="Use this tool when you want to get the results of an online web search"
    )

    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)

    python_repl = PythonREPLTool()
    
    # ========== ADD ACADEMIC TOOLS ==========
    
    citation_analysis_tool = Tool(
        name="analyze_citations_by_year",
        func=analyze_citations_by_year,
        description="""Analyze citations for a specific academic paper and sort by publication year in descending order. 
        Input should be the exact paper title. This tool will find papers that cite the given paper and organize them by year.
        Note: This process may take several minutes due to API rate limiting.
        Example usage: 'DeepLOB: Deep Convolutional Neural Networks for Limit Order Books'"""
    )
    
    paper_search_tool = Tool(
        name="search_academic_paper",
        func=search_academic_paper,
        description="""Search for detailed information about a specific academic paper by title. 
        Returns comprehensive details including authors, year, citations, abstract, etc.
        Input should be the paper title."""
    )
    
    author_search_tool = Tool(
        name="search_papers_by_author", 
        func=search_papers_by_author,
        description="""Search for papers by a specific author name. 
        Returns author information and their recent publications.
        Input should be the author's name."""
    )
    
    # Return all tools including the new academic ones
    return file_tools + [
        push_tool, 
        tool_search, 
        python_repl, 
        wiki_tool,
        citation_analysis_tool,
        paper_search_tool, 
        author_search_tool
    ]