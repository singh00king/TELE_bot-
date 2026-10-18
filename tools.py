import pickle
import pandas as pd

def predict_kyphosis(age:int, number:int, start:int)-> int:
    """ This function takes age, number, and start as input and returns the predicted kyphosis class (0 or 1) using the trained model. """
    
    model = pickle.load(open("model.pkl", "rb"))
    preprocessing = pickle.load(open("preprocessing.pkl", "rb"))
    patient = pd.DataFrame({"Age": [age], "Number": [number], "Start": [start]})
    patient[['Age']] = preprocessing["age_scaler"].transform(patient[['Age']])
    prediction = model.predict(patient)
    return preprocessing["label_encoder"].inverse_transform(prediction)[0]

from rag import search

def search_private_documents(query: str) -> str:
    """Use this tool to search through the user's private documents, PDFs, text files, and images.
    Always use this tool if the user asks a question about their files, documents, or data.
    Input should be a clear search query."""

    try:
        #RETRIEVAL
        # Call the search function from rag.py
        results = search(query, k=4)

        if not results:
            return "No relevant information found in the documents."

        #AUGMENTATION
        # Format the retrieved chunks into a single string for the LLM to read
        formatted_results = []
        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "Unknown file")
            content = doc.page_content
            formatted_results.append(f"--- Document {i+1} (Source: {source}) ---\n{content}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"An error occurred while searching the documents: {str(e)}"
    