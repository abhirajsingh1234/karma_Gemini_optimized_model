#import dependencies


import google.generativeai as genai
import os
from dotenv import load_dotenv
import gradio
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("vector_embeddings")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
history=[]

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8190,
  "response_mime_type": "text/plain",
}

retrieve_grader_1 = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
  system_instruction="""You are a grader assessing the relevance of a retrieved document to a user question. 
                        Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
                        Provide the binary score as JSON with a single key 'score'.
                        input format is 'question : question , document : document'. """
)
web_search_1_5= genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
  system_instruction="""You are an AI assistant specialized in retrieving spiritual and positive context from the Garud Puran.  
                        Use the given question to extract a meaningful and relevant context that aligns with the teachings of the Garud Puran.  
                        
                        If the user asks about their **karma, actions, or deeds**, retrieve context that explains both the **pros and cons** based on what will happen in **Swarg (heaven) and Nark (hell)** according to the Garud Puran and Sanatan Dharma.  
                        Ensure the context clearly describes the **specific rewards in Swarg** for good deeds, bringing peace and happiness, and the **specific punishments in Nark** for bad deeds, leading to suffering and atonement.  
                        The context should also highlight the **emotional and spiritual consequences** of one's actions, helping the user understand the **joy behind righteousness** and the **pain behind sinful acts**.  
                        
                        For **general queries**, retrieve an example from the Garud Puran that illustrates the concept in a way that makes it relatable and insightful.  
                        
                        Ensure the context is spiritually uplifting, guiding the user toward self-reflection and improvement, while keeping it concise (maximum of three paragraphs).  
                        Each retrieved context should be **unique**, providing fresh insights or varying perspectives while staying true to the scripture's teachings.  
                        Use **simple and clear English**, avoiding complex words, so the response is easy to understand for all users.  

                        """
)
answer_generator_2 = genai.GenerativeModel(
  model_name="gemini-2.0-pro-exp-02-05",
  generation_config=generation_config,
  system_instruction= """You are an AI assistant designed for question-answering tasks based on the Garud Puran.
                        
                        Use the retrieved context to generate an accurate and spiritually meaningful response.  
                        Ensure that the answer aligns with the teachings of the Garud Puran and maintains a positive and enlightening tone.  
                        
                        If the user asks about their **karma, actions, or deeds**, include both the **pros and cons** based on what will happen in **Swarg (heaven) and Nark (hell)** according to the Garud Puran and Sanatan Dharma.
                        Clearly mention the **specific rewards in Swarg** for good deeds and the **specific punishments in Nark** for bad deeds, as described in the scriptures.  

                        For **general queries**, provide a relevant example from the Garud Puran to illustrate the concept effectively.

                        ### **Response Structure:**  
                        1️⃣ **First, directly answer the question.**  
                           - Describe the **specific punishments in Nark (hell)** or **specific rewards in Swarg (heaven)** based on their karma, as per the Garud Puran.  
                           - Help the user understand the **joy and blessings** their good deeds bring and the **pain and suffering** caused by their bad deeds.  
                        
                        2️⃣ **Then, provide guidance on how to resolve or approach the problem from a spiritual perspective.**  
                           - Explain how the user can **overcome negative karma** through righteous actions, devotion, and self-correction.  
                           - Offer spiritual remedies or practices from the Garud Puran to seek divine grace and move toward a better path.  
                        
                        3️⃣ **Finally, include a general spiritual thought or insight related to the question.**  
                           - Inspire the user with a **positive message about karma** and the cycle of cause and effect.  
                           - Reinforce that **good deeds lead to inner peace and happiness**, while **wrong actions create suffering, but redemption is always possible through wisdom and self-improvement**.  
 
                        
                        Keep your response concise, with a maximum of three sentences.  
                        End with a positive thought related to the question, inspired by the spiritual wisdom of the Garud Puran.  
                        Use **simple and clear English**, avoiding complex words, so the response is easy to understand for all users.
                        **Respond in the same language in which the user asks the question.**  
                        """
  # system_instruction="""You are an AI assistant designed for question-answering tasks based on the Garud Puran.  

  #                       Use the retrieved context to generate an accurate and spiritually meaningful response.  
  #                       Ensure that the answer aligns with the teachings of the Garud Puran and maintains a positive and enlightening tone.  
                        
  #                       ### **Handling Different Types of Questions:**  
  #                       🟢 **For karma-related queries**  
  #                       - Include both the **pros and cons** of their actions based on **Swarg (heaven) and Nark (hell)** as per the Garud Puran and Sanatan Dharma.  
  #                       - Clearly describe **specific rewards in Swarg** for good deeds and **specific punishments in Nark** for bad deeds.  
                        
  #                       🟢 **For general queries about the Garud Puran**  
  #                       - Provide an **explanation or summary of relevant teachings** from the scripture.  
  #                       - If possible, include **a related story, parable, or example** from the Garud Puran to illustrate the concept effectively.  
  #                       - Ensure that the response remains **uplifting, thought-provoking, and spiritually insightful**.  
                        
  #                       ### **Response Structure:**  
  #                       1️⃣ **First, directly answer the question.**  
  #                          - If the question is **karma-related**, explain the **Swarg/Nark consequences**.  
  #                          - If the question is **general**, provide a **relevant explanation or story** from the Garud Puran.  

  #                       2️⃣ **Then, offer guidance or wisdom related to the topic.**  
  #                          - Provide insights into how the user can apply the knowledge in their life.  
  #                          - If the question is about karma, suggest ways to **correct negative karma**.  
  #                          - If it is a general query, explain **the deeper meaning or lesson** behind the concept.  
                        
  #                       3️⃣ **Finally, include a general spiritual thought or insight.**  
  #                          - Inspire the user with a **positive message** from the Garud Puran.  
  #                          - Reinforce **moral, ethical, and spiritual values** in a way that resonates with them.  
                        
                        # Keep your response concise, with a maximum of three sentences.  
                        # End with a positive thought related to the question, inspired by the spiritual wisdom of the Garud Puran.  
                        # Use **simple and clear English**, avoiding complex words, so the response is easy to understand for all users.  
                        # **Respond in the same language in which the user asks the question.**  
                        # """
  
)
hallucination_detection_3 = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
  system_instruction="""You are verifying whether the model-generated answer is factually correct based on the provided context.  
                        If the answer includes information not found in the context, classify it as hallucinated.  
                        Respond with a JSON object containing a single key `"hallucination"`, with a value of `"yes"` or `"no"`.  
                        
                        Output Format:  
                        {
                          "hallucination": "yes"  // If the answer contains hallucinated information  
                        }  
                        {
                          "hallucination": "no"   // If the answer is fully supported by the context  
                        }   """
)
question_resolving_detection_4 = genai.GenerativeModel(
  model_name="gemini-1.5-flash-8b",
  generation_config=generation_config,
  system_instruction="""You are a grader evaluating whether an answer is useful in resolving the given question.  
                        Assess if the answer is relevant, clear, and provides sufficient information to address the question.  
                        Respond with a JSON object containing a single key `"score"`, with a value of `"yes"` or `"no"`.  
                        
                        Input Format:  
                        question: {question}, answer: {answer}  
                        
                        Output Format:  
                        {
                          "score": "yes"  // If the answer is useful  
                        }  
                        {
                          "score": "no"   // If the answer is not useful  
                        }  
                        """
)

def retrieve_context(question, top_k=3):
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    question_embedding = embedding_fn([question])[0]

    # Retrieve top-k matching documents
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )
    if results["documents"]:
        # print(results["documents"])
        flat_documents = [doc for sublist in results["documents"] for doc in sublist]
        return " ".join(flat_documents) if flat_documents else "No relevant context found."
    
    return "No relevant context found."

def retrieve_grader_function(question):
    
    chat_session = retrieve_grader_1.start_chat(
                history=history
            )
    
    response = chat_session.send_message(question)
    
    model_response=response.text
    return model_response

# Web search function 
def web_search(query, num_results=3):
    
    chat_session = web_search_1_5.start_chat(
                history=history
            )
    
    response = chat_session.send_message(query)
    
    model_response=response.text
    return model_response
    

def answer_generator_function(question):
    
    chat_session = answer_generator_2.start_chat(
                history=history
            )
    
    response = chat_session.send_message(question)
    
    model_response=response.text
    return model_response
    
def hallucination_detection_function(question):
    
    chat_session = hallucination_detection_3.start_chat(
                history=history
            )
    
    response = chat_session.send_message(question)
    
    model_response=response.text
    return model_response
def question_resolving_detection_function(question):
    
    chat_session = question_resolving_detection_4.start_chat(
                history=history
            )
    
    response = chat_session.send_message(question)
    
    model_response=response.text
    return model_response


#Full Path
def Full_Flow(question):
    flag=0
    while True:
        if flag==0:
            document = retrieve_context(question)
            model_input = f"question : {question} , document : {document}"
            output = retrieve_grader_function(model_input)
            
            if 'yes' in output:
                print('document found in database')
            elif 'no' in output:
                print('searching web.....')
                print('document found on web.....')
                document = web_search(question)
        elif flag==1:
            print('searching web.....')
            print('document found on web.....')
            document = web_search(question)
        # print(document)
        while True:   
            #Generation of answer based on context
            model_input = f"question : {question},context : {document}"
            answer = answer_generator_function(model_input)
            print('answer fetched from document')
        
            #Hallucination detection to check the correctness of answer
            hallucination_check_input = f"context : {document}, answer : {answer}"
            hallucination_output = hallucination_detection_function(hallucination_check_input)
            if 'yes' in hallucination_output:
                print('hallucination detected.')
                print('regenerating the answer....\n')
                continue
            elif 'no' in hallucination_output:
                print('no hallucination detected')
                question_resolver_input = f' question: {question}, answer: {answer}'
                question_resolver_output = question_resolving_detection_function(question_resolver_input)
                # print(question_resolver_output)
                break
            else: return None
        if 'no' in question_resolver_output:
            print('the generated answer do not resolve the query\n')
            print('searching the relevant document again.....\n')
            flag=1
            continue
            
        elif 'yes' in question_resolver_output:
            print('generated answer will resolve the query\n\n')
            return 'answer :'+answer
        else: return None

gradio_ui = gradio.Interface(
    fn=Full_Flow,
    inputs=gradio.Textbox(label="Ask a Question"),
    outputs=gradio.Markdown(label="Response"),
    title="Karma AI - garud puran",
    description="Ask a question about ur spiritual thoughts and curosity."
)

# Launch Gradio app
gradio_ui.launch()