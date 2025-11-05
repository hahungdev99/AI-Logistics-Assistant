# 📦 Hermes AI Logistics Assistant

An intelligent AI-powered logistics assistant built with LangChain ReAct Agent, GPT-4o, and Streamlit. Hermes helps operations managers analyze shipment data, detect trends, and make predictions using natural language queries.

## 🌟 Features

### 🔍 **Intelligent Query Understanding**

- Natural language processing for logistics questions
- Multi-step reasoning with ReAct Agent pattern
- Context-aware responses with memory

### 🛠️ **10 Specialized Tools**

1. **Overall Statistics** - Get comprehensive logistics overview
2. **Route Analysis** - Find most delayed routes by period
3. **Warehouse Ranking** - Compare warehouse performance
4. **Delay Breakdown** - Analyze delays by reason
5. **Advanced Filtering** - Query shipments with multiple criteria
6. **Delivery Time Analysis** - Calculate average delivery metrics
7. **On-Time Rate Calculation** - Track delivery performance
8. **Trend Detection** - Identify increasing/decreasing patterns
9. **ML Predictions** - Forecast next week's delays
10. **Optimization Recommendations** - Get actionable improvement suggestions

### 📊 **Data Analytics**

- Time-series analysis
- Statistical aggregations
- Trend detection
- Performance comparisons

### 🔮 **Machine Learning**

- Linear regression for delay predictions
- Historical pattern analysis
- Confidence scoring

### 💬 **User-Friendly Interface**

- Clean Streamlit web interface
- Chat-based interaction
- Conversation history
- Example queries
- Real-time responses

## 🏗️ Architecture

```
hermes-logistics-assistant/
├── data/
│   └── shipments.csv              # Mock shipment data (70 records)
├── backend/
│   ├── __init__.py
│   ├── agent.py                   # LangChain ReAct Agent
│   ├── tools.py                   # 10 specialized tools
│   └── data_processor.py          # Data handling & preprocessing
├── frontend/
│   └── app.py                     # Streamlit web interface
├── requirements.txt
├── .env.example
└── README.md
```

### **Technology Stack**

- **AI Framework**: LangChain (ReAct Agent)
- **LLM**: OpenAI GPT-4o
- **ML**: Scikit-learn (Linear Regression)
- **Data**: Pandas, NumPy
- **Frontend**: Streamlit
- **Language**: Python 3.9+

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key

### Installation

1. **Clone or download the project**

```bash
cd hermes-logistics-assistant
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

5. **Verify data file exists**

```bash
# Make sure data/shipments.csv exists
ls data/shipments.csv
```

### Running the Application

**Start Streamlit frontend:**

```bash
streamlit run frontend/app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Initialize Agent**

   - Enter your OpenAI API key in the sidebar
   - Click "🚀 Initialize Agent"
   - Wait for confirmation and data summary

2. **Start Chatting**

   - Type questions in the chat input
   - Or click example queries in the sidebar
   - View responses in the chat interface

3. **Example Queries**
   - "Which route had the most delays last week?"
   - "Show the top 3 warehouses with highest processing time"
   - "Predict the average delay for next week"
   - "What is the on-time delivery rate this week?"
   - "Give me warehouse optimization recommendations"

## 📊 Data Structure

### Shipments CSV Format

```csv
id, route, warehouse, delivery_time_days, delay_minutes, delay_reason, date, processing_time_hours, shipment_value, weight_kg, destination_city, on_time
```

### Fields Explanation

- **id**: Unique shipment identifier
- **route**: Route name (Route A, Route B, Route C)
- **warehouse**: Warehouse code (WH1, WH2, WH3)
- **delivery_time_days**: Total delivery time in days
- **delay_minutes**: Delay duration (0 = on-time)
- **delay_reason**: Cause of delay (Weather, Traffic, Mechanical, Customs, None)
- **date**: Shipment date
- **processing_time_hours**: Warehouse processing time
- **shipment_value**: Monetary value
- **weight_kg**: Shipment weight
- **destination_city**: Delivery destination
- **on_time**: Yes/No flag

### Sample Data

The included `shipments.csv` contains 70 mock records spanning October-November 2024.

## 🤖 Agent Capabilities

### ReAct Agent Pattern

The agent uses the ReAct (Reasoning + Acting) pattern:

1. **Thought**: Analyzes the question
2. **Action**: Selects appropriate tool(s)
3. **Observation**: Processes tool results
4. **Repeat**: Can call multiple tools sequentially
5. **Final Answer**: Synthesizes insights

### Multi-Tool Orchestration

For complex queries, the agent can:

- Call multiple tools in sequence
- Use output from one tool as input to another
- Combine results from different tools
- Make intelligent decisions about which tools to use

### Example: Prediction Workflow

```
User: "Predict delays and give recommendations"

Agent Flow:
1. Tool: get_delay_trend_analysis() → Gets historical patterns
2. Tool: predict_next_week_delay() → Makes prediction
3. Tool: get_warehouse_optimization_recommendation() → Suggests actions
4. Final Answer: Synthesized response with all insights
```

## 📈 Query Examples

### Basic Queries

```
"What was the average delay in October?"
"How many shipments were delayed?"
"Show me all routes"
```

### Advanced Queries

```
"Which warehouse has the longest processing time and what can we do to improve it?"
"Compare delay trends between last week and this week"
"Predict next week's delays and explain the confidence level"
```

### Vietnamese Queries (Translated to English internally)

```
Input: "Tổng số lô hàng bị delay trong tháng 10 là bao nhiêu?"
→ Agent processes in English: "How many shipments were delayed in October?"
```

## 🎯 Evaluation Metrics

### 1. Accuracy

- Correct data retrieval from CSV
- Accurate calculations and aggregations
- Proper filtering and date handling

### 2. Explainability

- Tool usage is logged and visible
- Data sources are traceable
- Reasoning steps are transparent

### 3. Response Time

- Average query time: 3-8 seconds
- Tool calls: 1-5 per query
- Caching for repeated queries

### 4. Coverage

- Supports 10+ query types
- Handles ambiguous questions
- Multi-step reasoning capability

## 🔧 Customization

### Adding New Tools

1. Create tool function in `backend/tools.py`

```python
@tool
def your_new_tool(param: str) -> str:
    """Tool description for LLM"""
    # Your logic here
    return json.dumps(result)
```

2. Add to `ALL_TOOLS` list

```python
ALL_TOOLS = [..., your_new_tool]
```

### Modifying Data Structure

1. Update CSV in `data/shipments.csv`
2. Adjust `DataProcessor` methods if needed
3. Update tool functions to handle new fields

### Changing LLM Model

Edit `backend/agent.py`:

```python
self.llm = ChatOpenAI(
    model="gpt-4o-mini",  # or any OpenAI model
    temperature=0,
)
```

## 🐛 Troubleshooting

### Common Issues

**Agent not initializing**

- Check OpenAI API key is valid
- Verify internet connection
- Ensure sufficient API credits

**Data file not found**

- Verify `data/shipments.csv` exists
- Check file path in sidebar config
- Ensure CSV format is correct

**Slow responses**

- GPT-4o is powerful but can be slower
- Consider using `gpt-4o-mini` for faster responses
- Check OpenAI API status

**Import errors**

- Reinstall requirements: `pip install -r requirements.txt`
- Verify virtual environment is activated
- Check Python version (3.9+)

## 📝 Project Structure Details

### Backend Components

**agent.py**

- Initializes LangChain ReAct Agent
- Configures GPT-4o LLM
- Manages conversation memory
- Handles tool orchestration

**tools.py**

- 10 specialized tools
- JSON-formatted responses
- Error handling
- Data validation

**data_processor.py**

- CSV loading and preprocessing
- Date filtering utilities
- Statistical calculations
- Aggregation functions

### Frontend Components

**app.py**

- Streamlit web interface
- Chat UI with history
- Sidebar configuration
- Example queries
- Custom CSS styling

## 🎓 Learning Resources

### LangChain ReAct Agent

- [LangChain Documentation](https://python.langchain.com/docs/modules/agents/)
- [ReAct Pattern Paper](https://arxiv.org/abs/2210.03629)

### OpenAI GPT-4o

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [GPT-4o Capabilities](https://openai.com/research/gpt-4o)

## 🤝 Contributing

Suggestions for improvements:

1. Add more sophisticated ML models
2. Implement caching for repeated queries
3. Add data visualization charts
4. Support multiple languages
5. Implement user authentication
6. Add export functionality

## 📄 License

This project is created for educational and demonstration purposes.

## 🙏 Acknowledgments

- Built with LangChain
- Powered by OpenAI GPT-4o
- UI framework: Streamlit
- Data processing: Pandas

---

**Made with ❤️ for Operations Managers**

For questions or issues, please refer to the documentation or check tool implementations in the code.
