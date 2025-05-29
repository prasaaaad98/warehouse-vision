# Warehouse Vision AI

An intelligent warehouse management system powered by AI that automates inventory tracking, quality control, and product information extraction.

## 🌟 Features

- **Brand Recognition**: Automatically identify products and brands with quantity tracking
- **Freshness Detection**: Real-time analysis of produce quality with freshness scoring
- **Smart Item Counting**: Automated inventory counting with location tracking
- **Nutrition Information Extraction**: Automated reading of nutrition labels
- **MRP & Expiry Management**: Automated date detection and shelf life calculation

## 🚀 Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React.js
- **AI**: Google Gemini 2.0 Flash
- **Image Processing**: OpenCV, PIL
- **API**: RESTful architecture

## 📋 Prerequisites

- Python 3.8+
- Node.js and npm


## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/prasaaaad98/warehouse-vision.git
cd warehouse-vision
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Set up the frontend:
```bash
cd frontend
npm install
```

## 🚀 Running the Application

1. Start the backend server:
```bash
cd backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at:
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000

## 📚 API Endpoints

- `POST /api/brand-recognition`: Identify brands and products
- `POST /api/freshness-detection`: Analyze produce freshness
- `POST /api/item-counting`: Count items in warehouse images
- `POST /api/nutrition-info`: Extract nutrition information
- `POST /api/mrp-expiry`: Detect MRP and expiry dates

## 🔒 Security

- API keys are stored in environment variables
- CORS is configured for specific origins
- Input validation for all endpoints
- Error handling and logging

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Prasad Jore - Yash Gupta

