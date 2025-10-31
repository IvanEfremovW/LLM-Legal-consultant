import gradio as gr

from .rag_pipeline import rag_chain


def pipeline(text: str) -> str:
    return rag_chain.invoke(text)


def main():
    interface = gr.Interface(
        fn=pipeline,
        inputs=gr.TextArea(label="Введите вопрос"),
        outputs=gr.Textbox(label="Ответ", lines=25),
    )

    interface.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
