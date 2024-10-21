Project: EmotiCare AI - Mental Health Chatbot
Project Overview:
MINDCARE AI is a state-of-the-art mental health chatbot developed using a Retrieve and Generate (RAG) approach. The chatbot leverages advanced natural language processing (NLP) techniques to provide users with personalized mental health support and guidance. By combining a curated dataset with powerful AI models, MINDCARE AI offers real-time, context-aware responses to user inquiries about mental well-being.

Dataset Features:
The primary dataset utilized for training and response generation in MINDCARE AI is the Marmik Pandya Mental Health Dataset. This dataset is specifically designed to address various mental health concerns, offering a rich resource for developing the chatbot's capabilities.

Total Number of Records: The dataset contains approximately 4,000 records, each comprising an input-output pair that addresses different mental health topics.
Features:
Input Text: This feature includes diverse mental health queries and concerns, such as anxiety, stress management, depression, and coping strategies.
Output Text: Each input is paired with a professionally crafted response, providing users with relevant advice and insights.
Tags: Certain records may include tags or categories to facilitate targeted responses based on specific mental health issues.
Emotional Tone: The dataset captures various emotional tones, allowing the chatbot to respond empathetically and appropriately to user inquiries.
Project Process:
Data Preparation:

The dataset was loaded and preprocessed to ensure consistency and clarity in the text. This included text normalization, removing special characters, and preparing the data for embedding.
Embedding and Indexing:

The SentenceTransformer model was used to generate embeddings for both input and output texts. These embeddings were then indexed using Pinecone, facilitating efficient data retrieval based on user queries.
Chatbot Development:

The user interface was developed using Streamlit, creating a responsive and interactive environment for users to engage with the chatbot.
The chatbot was designed to process user input, query the Pinecone database for relevant responses, and utilize Groq’s Llama 3.2 model to generate contextual replies when no direct match was found.
Response Mechanism:

Upon receiving user input, the chatbot retrieves relevant data from the dataset or generates an appropriate response using the Llama model.
Users are presented with options to either engage in further conversation or receive concise solutions to their issues.
User Experience Testing:

The chatbot interface was tested with real users to refine interactions, enhance response quality, and ensure that users felt supported and understood during their conversations.
Deployment:

The finalized chatbot was delivered to the client on Fiverr, showcasing a seamless integration of data retrieval and AI response generation that enhances the overall user experience.
Conclusion:
MINDCARE AI stands as a testament to the power of combining AI technology with mental health resources. By utilizing a well-curated dataset and advanced NLP techniques, the chatbot provides users with timely and personalized mental health support. The user-friendly interface invites users to explore their concerns in a safe and supportive environment.

Interface Preview:
