import gradio as gr
import asyncio
from sidekick import Sidekick

class UI:
    def __init__(self):
        self.sidekick = None
        self.history = []
        self.success_criteria = "Provide a helpful and accurate response to the user's query."

    async def setup(self):
        # Initialize the Sidekick
        self.sidekick = Sidekick()
        await self.sidekick.setup()
        return self.sidekick

    async def respond(self, message, criteria, history):
        if not self.sidekick:
            await self.setup()
        
        # Update success criteria if provided
        if criteria:
            self.success_criteria = criteria
        
        # Chuyển đổi tin nhắn thành định dạng mà Sidekick có thể hiểu
        if isinstance(message, str):
            message_obj = message
        else:
            message_obj = message
        
        # Gọi Sidekick để xử lý tin nhắn
        result = await self.sidekick.run_superstep(message_obj, self.success_criteria, [])
        
        # Lấy tin nhắn người dùng và phản hồi của assistant
        user_msg = message
        bot_msg = result[-2]["content"]  # Lấy phản hồi của assistant (bỏ qua feedback)
        
        # Thêm tin nhắn mới vào lịch sử Gradio
        history.append([user_msg, bot_msg])
        
        return "", history, self.success_criteria

    def launch(self, inbrowser=True):
        # Create Gradio interface with modern theme
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")) as demo:
            with gr.Row():
                gr.Markdown("""
                # Agentic AI
                """, elem_id="app-title")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(height=600, elem_id="chatbox")
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask me anything...", 
                            label="Your message",
                            scale=8,
                            container=False
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        criteria = gr.Textbox(
                            placeholder="Enter success criteria for the AI response...",
                            label="Success Criteria",
                            value=self.success_criteria,
                            lines=2
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### Options
                    """)
                    clear = gr.Button("Clear Chat", variant="secondary")
                    gr.Markdown("""
                    ### About
                    Agentic AI is an advanced assistant that can help you with browsing, searching, and more!
                    
                    Set specific criteria to guide how the AI should respond to your queries.
                    """)
            
            # Event handlers
            submit_btn.click(self.respond, [msg, criteria, chatbot], [msg, chatbot, criteria])
            msg.submit(self.respond, [msg, criteria, chatbot], [msg, chatbot, criteria])
            clear.click(lambda: (None, [], self.success_criteria), None, [msg, chatbot, criteria], queue=False)
        
        # Launch the interface
        demo.queue().launch(inbrowser=inbrowser, share=True)

# Create a singleton instance
ui = UI()