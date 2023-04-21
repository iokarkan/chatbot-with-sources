import gradio as gr
from pathlib import Path
from chatbot.backend import ChatbotBackend

####################################################

with gr.Blocks() as demo:
    chatbot_instance = gr.State(ChatbotBackend())
    api_key_instance = gr.State()
    markdown_sources = gr.State()

    def set_api(api_key, history, chatbot_instance):
        chatbot_instance.authenticate(api_key)
        if chatbot_instance.llm is not None:
            history = history + [[None, "API key set successfully."]]
        else:
            history = history + [[None, "Invalid OpenAI key..."]]
        return history, chatbot_instance

    def add_text(history, text, chatbot_instance):
        history = history + [(text, None)]
        return (
            history,
            "",
            chatbot_instance,
        )  # "" is meant to clear the textbox input message

    def add_file(history, file, chatbot_instance, markdown_sources):
        if chatbot_instance.llm is not None:
            chatbot_instance.update_sources(fname=file.name)
            history = history + [
                (
                    f"Uploaded file {file.name}",
                    f"**File {Path(file.name).stem}{Path(file.name).suffix} received! You can now query information regarding this source!**",
                )
            ]
            markdown_sources = (
                "# Loaded Text Sources\n\n"
                + "\n".join(chatbot_instance.markdown_sources)
                + "\n\n### NB: The same file isn't accepted twice in a row."
            )
        else:
            history = history + [
                (
                    f"Uploaded file {file.name}",
                    "File not processed. " + "Please paste your OpenAI API key...",
                )
            ]
        return history, chatbot_instance, markdown_sources

    def bot(history, chatbot_instance):
        # bot needs to update the "answer" part of the last conversation step, aka history[-1]
        user_input = history[-1][0]
        # answer = "Code me scrub!"
        if chatbot_instance.llm is not None:
            answer = chatbot_instance.generate_response(user_input, history[-3:-1])
        else:
            answer = "Please paste your OpenAI API key..."
        history[-1][1] = answer
        return history, chatbot_instance

    with gr.Row():
        with gr.Column(scale=0.5):
            chatbot_output = gr.Chatbot(
                value=[(None, "I'm the database Chatbot 🤖 ! What is your request?")],
                elem_id="chatbot",
            ).style(height=500)
            with gr.Row():
                with gr.Column(scale=0.8):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter, or upload an .txt or .csv file",
                    ).style(container=False)
                with gr.Column(scale=0.1, min_width=0):
                    chat_btn = gr.Button("Chat")
                with gr.Column(scale=0.2, min_width=0):
                    upload_btn = gr.UploadButton("📁", file_types=["text"])
        with gr.Column(scale=0.5):
            with gr.Row():
                with gr.Column():
                    openai_api_key_textbox = gr.Textbox(
                        placeholder="Paste your OpenAI API key...",
                        show_label=False,
                        lines=1,
                        type="password",
                    ).style(container=False)
                with gr.Column(scale=0.2, min_width=0):
                    api_btn = gr.Button("Set")
            markdown = gr.Markdown(
                "# Loaded Text Sources\n\nCurrently empty...\n\nPlease paste an API key before uploading a file."
            )

    api_btn.click(
        fn=set_api,
        inputs=[openai_api_key_textbox, chatbot_output, chatbot_instance],
        outputs=[chatbot_output, chatbot_instance],
    )

    txt.submit(
        fn=add_text,
        inputs=[chatbot_output, txt, chatbot_instance],
        outputs=[chatbot_output, txt, chatbot_instance],
    ).then(
        fn=bot,
        inputs=[chatbot_output, chatbot_instance],
        outputs=[chatbot_output, chatbot_instance],
    )

    chat_btn.click(
        fn=add_text,
        inputs=[chatbot_output, txt, chatbot_instance],
        outputs=[chatbot_output, txt, chatbot_instance],
    ).then(
        fn=bot,
        inputs=[chatbot_output, chatbot_instance],
        outputs=[chatbot_output, chatbot_instance],
    )
    
    upload_btn.upload(
        fn=add_file,
        inputs=[chatbot_output, upload_btn, chatbot_instance, markdown],
        outputs=[chatbot_output, chatbot_instance, markdown],
    )

demo.launch()
