import qa_functions as qa
import gradio as gr

choices = [("The Theory that Would Not Die", "theory_that_would_not_die"),
           ("Writing with Style", "writing_with_style"),
           ("Trustworthy Online Controlled Experiments", "trustworthy_online_controlled_experiments")]

with gr.Blocks(theme=gr.themes.Default(primary_hue="orange", secondary_hue="zinc")) as gradio_ui:
    gr.Markdown('<h1 align = "center"> PDF Query Tool! </h1>')
    with gr.Tabs() as tabs:
        with gr.Tab("Dataset", id=0):
            org_name = gr.Text(label="DeepLake Dataset Organization Name")
            dataset_name = gr.Dropdown(choices, label="Dataset Name")
            open_ai_key = gr.Text(label="OpenAI Key")
            deeplake_key = gr.Text(label="DeepLake Key")
            btn0 = gr.Button("Submit")
            success = gr.Text(show_label=False)
            mq_retriever = gr.State()
            chain = gr.State()

            (btn0.click(qa.initialize_retrievers,
                        inputs=[org_name, dataset_name,
                                open_ai_key, deeplake_key],
                        outputs=[mq_retriever, chain, success])
             .then(lambda: gr.Tabs(selected=1), None, tabs))

        with gr.Tab("Chatbot", id=1):
            chatbot = gr.Chatbot(value=[], elem_id="chatbot")
            with gr.Column(scale=85):
                human_input = gr.Textbox(
                    show_label=False, placeholder="Enter your question here")
            btn1 = gr.Button("Submit")
            btn2 = gr.Button("Clear")
            source = gr.Text(label="Sources")
            btn1.click(qa.chat, inputs=[
                       human_input, mq_retriever, chain, chatbot], outputs=[human_input, chatbot, source])
            btn2.click(lambda: None, None, chatbot, queue=False)
            btn2.click(lambda: None, None, source, queue=False)

if __name__ == '__main__':
    gradio_ui.launch()
