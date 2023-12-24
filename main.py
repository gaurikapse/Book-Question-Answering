import qa_functions as qa
import gradio as gr

choices = [("The Theory that Would Not Die", "theory_that_would_not_die"),
           ("Writing with Style", "writing_with_style"),
           ("Trustworthy Online Controlled Experiments", "trustworthy_online_controlled_experiments")]


theme = theme = (gr.themes.Default(
    primary_hue="pink",
    secondary_hue="slate",
    neutral_hue='sky')
    .set(button_shadow_active='*shadow_inset',
         button_shadow_hover='*shadow_drop_lg',
         button_primary_background_fill='linear-gradient(180deg, *primary_50 0%, *primary_200 50%, *primary_300 50%, *primary_200 100%)',
         button_secondary_background_fill='linear-gradient(180deg, *neutral_50 0%, *neutral_200 50%, *neutral_300 50%, *neutral_200 100%)'))


with gr.Blocks(theme=theme) as gradio_ui:
    gr.Markdown('<h1 align="center"> PDF Query Tool! </h1>')
    with gr.Tabs() as tabs:
        with gr.Tab("Dataset", id=0):
            dataset_name = gr.Dropdown(choices, label="Dataset Name")
            open_ai_key = gr.Text(label="OpenAI Key")
            deeplake_key = gr.Text(label="DeepLake Key")
            btn0 = gr.Button("Submit", variant="primary")
            success = gr.Text(show_label=False)
            mq_retriever = gr.State()
            chain = gr.State()

            (btn0.click(qa.initialize_retrievers,
                        inputs=[dataset_name,
                                open_ai_key, deeplake_key],
                        outputs=[mq_retriever, chain, success])
             .then(lambda: gr.Tabs(selected=1), None, tabs))

        with gr.Tab("Chatbot", id=1):
            chatbot = gr.Chatbot(value=[], elem_id="chatbot")
            human_input = gr.Textbox(
                show_label=False, placeholder="Enter your question here")
            with gr.Row():
                btn1 = gr.Button("Submit", variant="primary")
                btn2 = gr.Button("Clear", variant="secondary")
            source = gr.Text(label="Sources")
            btn1.click(qa.chat, inputs=[
                       human_input, mq_retriever, chain, chatbot], outputs=[human_input, chatbot, source])
            human_input.submit(qa.chat, inputs=[
                human_input, mq_retriever, chain, chatbot], outputs=[human_input, chatbot, source])
            btn2.click(lambda: None, None, chatbot, queue=False)
            btn2.click(lambda: None, None, source, queue=False)
            btn2.click(lambda: None, None, human_input, queue=False)

if __name__ == '__main__':
    gradio_ui.launch()
