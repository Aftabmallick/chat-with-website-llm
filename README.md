# Chat With Website Data

This project, "Chat With Website Data," is designed to facilitate natural language interaction with website content, extracting relevant data and insights through conversational means. By utilizing Streamlit, LangChain, LangChain-OpenAI, BeautifulSoup4, Python-Dotenv, and Pinecone Client, this tool aims to provide a seamless experience for users to query and analyze website data effortlessly.

## Features

- **Natural Language Interaction**: Users can engage with the tool using natural language queries, making the process intuitive and user-friendly.
  
- **Data Extraction**: The project employs BeautifulSoup4 to parse HTML content, extracting meaningful information from websites specified by the user.
  
- **Language Processing**: Through LangChain and LangChain-OpenAI, the tool processes and understands user queries, ensuring accurate and relevant responses.
  
- **Streamlit Integration**: The interface is built using Streamlit, enabling easy deployment and interaction with the tool through a web browser.
  
- **Pinecone Client Integration**: Pinecone Client is utilized for efficient similarity search and retrieval, enhancing the speed and accuracy of data analysis.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Aftabmallick/chat-with-website-llm.git
   ```

2. Navigate to the project directory:
   ```
   cd chat-with-website-llm
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure you have the necessary environment variables set up, especially if using external APIs. You can use `python-dotenv` to manage environment variables conveniently.

5. Run the Streamlit app:
   ```
   streamlit run src/app.py
   ```

## Usage

Once the Streamlit app is running, you can access it via your web browser. Interact with the tool by typing natural language queries into the provided input field. The tool will process your query, extract relevant data from the specified website(s), and display the results accordingly.

## API References

- **Streamlit**: The user interface and application logic are built using Streamlit, a powerful library for creating web applications with Python. Refer to the [Streamlit documentation](https://docs.streamlit.io/) for detailed information on usage and functionality.

- **LangChain and LangChain-OpenAI**: LangChain and LangChain-OpenAI are utilized for natural language processing and understanding. Check out their respective documentation on [LangChain](https://langchain.readthedocs.io/en/latest/) and [LangChain-OpenAI](https://github.com/langchain/langchain-openai) for implementation details and usage instructions.

- **BeautifulSoup4**: BeautifulSoup4 is used for web scraping and HTML parsing. Refer to the [BeautifulSoup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) for guidance on web scraping techniques and BeautifulSoup usage.

- **Pinecone Client**: Pinecone Client is employed for similarity search and retrieval. Visit the [Pinecone documentation](https://www.pinecone.io/docs/) for comprehensive information on integrating Pinecone with your applications.

## Deployment

This project is hosted on Streamlit Cloud for seamless deployment and accessibility. You can access the hosted version [here](https://chat-with-website.streamlit.app/).

## Contributing

Contributions to this project are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Be sure to follow the existing coding style and guidelines.


## Contact

For any inquiries or feedback, feel free to reach out to the project maintainer:

- **Name:** [Aftab Mallick](https://github.com/Aftabmallick)
- **Email:** [aftabmallick000@gmail.com](mailto:aftabmallick000@gmail.com)

We hope you find this tool useful and look forward to your contributions! Happy analyzing!
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://aftabmallick.github.io/MyWebsite/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aftab-mallick/)

