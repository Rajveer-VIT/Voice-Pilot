import speech_recognition as sr
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
import asyncio
from openai.helpers import LocalAudioPlayer
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MONGODB_URI = os.getenv("MONGODB_URI")

config = {"configurable": {"thread_id": "11"}}


def main():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)

        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            r.pause_threshold = 2

            while True:
                print("Say something!")
                audio = r.listen(source)

                print("Processing audio...")
                try:
                    sst = r.recognize_google(audio)
                    print("You Said:", sst)

                    for event in graph.stream(
                        {"messages": [{"role": "user", "content": sst}]},
                        config,
                        stream_mode="values"
                    ):
                        if "messages" in event:
                            event["messages"][-1].pretty_print()
                except sr.UnknownValueError:
                    print("Sorry, I could not understand your speech.")
                except sr.RequestError as e:
                    print(f"Speech Recognition error: {e}")


async def speak(text: str):
    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a cheerful and positive tone.",
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


if __name__ == "__main__":
    main()
    # To test speech output, uncomment below:
    # asyncio.run(speak("This is a sample voice. Hi Piyush"))
