# financial-analysis-with-llm

## AI-Powered Financial Analysis Tool

The application uses a combination of Language model, natural language processing (NLP), and financial analysis techniques to extract, process, and analyze data from uploaded financial documents (Excel format). It provides insights such as transaction summaries, suspicious trends, cash deposit detection, entity recognition, and potential ML/TF activities.


## Key Features

- **Financial Document Processing:** Upload and process Excel files containing transaction data.
- **AI-Powered Analysis:** Utilize OpenAI's language models to analyze transaction data.
- **Suspicious Activity Detection:** Identify potential suspicious financial trends.
- **Entity Recognition:** Extract and display entities like persons, organizations, and locations using NLP.
- **Money Laundering/Terrorism Financing Hypothesis Generation:** Generate hypotheses regarding ML/TF activities based on transaction data.
- **Interactive UI:** Streamlit interface for easy uploading and viewing of results.

## Technologies Used

- **Streamlit:** For building the web interface.
- **Pandas:** For data manipulation and analysis.
- **Spacy:** For natural language processing and entity recognition.
- **OpenAI API:** For accessing advanced AI language models.
- **LangChain:** For leveraging AI agents in processing and analyzing data.

## Installation and Usage

**Clone the Repository:**
```bash
git clone https://github.com/babdulhakim2/financial-analysis-with-llm.git
cd financial-analysis-with-llm

pip install -r requirements.txt

streamlit run app.py
```

Interact with the Application:

Navigate to the provided local URL (usually http://localhost:8501).
Upload a financial document in Excel format.
Explore various analytical features provided by the application.


