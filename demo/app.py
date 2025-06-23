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

    async def respond(self, message, history):
        if not self.sidekick:
            await self.setup()
        
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
        
        return history

    def launch(self, inbrowser=True):
        # Create Gradio interface
        with gr.Blocks() as demo:
            gr.Markdown("# Sidekick Assistant")
            gr.Markdown("Ask me anything and I'll help you with browsing, searching, and more!")
            
            chatbot = gr.Chatbot(height=600)
            msg = gr.Textbox(label="Your message")
            clear = gr.Button("Clear")
            
            msg.submit(self.respond, [msg, chatbot], chatbot)
            clear.click(lambda: None, None, chatbot, queue=False)
        
        # Launch the interface
        demo.queue().launch(inbrowser=inbrowser, share=True)

# Create a singleton instance
ui = UI()