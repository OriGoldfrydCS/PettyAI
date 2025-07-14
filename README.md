# PettyAI - Pet Adoption Prediction Application

A web application that helps animal shelters predict how quickly pets will be adopted based on their characteristics, health status, and other factors.

---

## Table of Contents

- [Application Screenshots](#️-application-screenshots)
- [Features](#features)
- [Prediction Categories](#prediction-categories)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Using the Application](#using-the-application)
- [Data Management](#data-management)
- [Customization](#customization)
- [Browser Compatibility](#browser-compatibility)
- [Data Privacy](#data-privacy)
- [Road Map](#road-map)
- [Contributing](#contributing)
- [License](#license)
- [Technical Implementation](#technical-implementation)
- [Getting Started](#getting-started)

---

## Application Screenshots

### Smart Dashboard
<div align="center">
  <img src="images/main_dashboard_with_pets.png" alt="PettyAI Dashboard with Pets" width="800">
  <p><em>Dashboard showing real-time statistics and recent pet profiles with predictions</em></p>
</div>

### Clean Start
<div align="center">
  <img src="images/main_dashboard_empty.png" alt="PettyAI Empty Dashboard" width="800">
  <p><em>Clean dashboard interface when starting fresh - ready for your first pet!</em></p>
</div>

### Pet Profile Creation
<div align="center">
  <img src="images/add_new_pet_profile.png" alt="Add New Pet Profile Form" width="800">
  <p><em>Intuitive 6-step form for creating comprehensive pet profiles</em></p>
</div>

### Interactive Pet Cards
<div align="center">
  <img src="images/pet_card.png" alt="Pet Card Detail View" width="800">
  <p><em>Detailed pet information cards with clickable modals for easy management</em></p>
</div>

---

## Features

- **Multi-step Form Interface**: Intuitive 6-step form for entering pet information
- **AI-Powered Predictions**: Uses AI algorithms to predict adoption timeframes based on pet characteristics
- **Smart Dashboard**: Real-time statistics, insights, and overview of all pets in the system
- **Interactive Pet Cards**: Click on any pet image to view detailed information and edit profiles
- **Photo Upload System**: Support for multiple pet photos with drag-and-drop interface

---

## Prediction Categories

The system predicts adoption time in 4 categories:
- **0-7 days**: Same day to first week (Fast adoption)
- **8-30 days**: First month (Fast adoption)
- **31-90 days**: 1-3 months (Moderate adoption)
- **100+ days**: Long-term care needed (Slow adoption)

---

## Project Structure

```
ML_FP_app - Final/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── backend.py                   # Main Flask server and ML backend
├── PROJECT_STRUCTURE.md         # Detailed structure documentation
│
├── frontend/                    # Frontend files
│   ├── local_app.html           # Main HTML application
│   └── assets/                  # Frontend assets
│       ├── styles.css           # Application styles
│       ├── script.js            # Main JavaScript logic
│       └── data.js              # Data constants and mappings
│
└── data/                        # Data and model files
    ├── clip_mlp_model.pt        # Trained PyTorch model
    └── pet_adoption_data.csv    # Pet data storage
```
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Setup Instructions

### Frontend Only (Quick Start)

1. **Open the Application**
   - Simply open `local_app.html` in a web browser
   - The dashboard will load by default showing any saved pets
   - No server setup required for basic functionality
   - The app will work with local storage and simulated predictions

### Full Setup with Python Backend

1. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Backend Server**
   ```bash
   python backend.py
   ```

3. **Open the Frontend**
   - Open `local_app.html` in a web browser
   - The frontend will automatically connect to the backend API

---

## Using the Application

### Adding a New Pet Profile

1. **Basic Information** (Step 1)
   - Select pet type (Dog/Cat)
   - Enter name (optional)
   - Enter age in months
   - Set quantity (for litters)
   - Set adoption fee

2. **Physical Characteristics** (Step 2)
   - Select breed from dropdown (options appear after selecting pet type in Step 1)
   - Choose gender
   - Select primary color
   - Set expected size when fully grown
   - Choose fur length

3. **Health Information** (Step 3)
   - Vaccination status
   - Deworming status
   - Sterilization status
   - Overall health condition

4. **Location** (Step 4)
   - Select state

5. **Description** (Step 5)
   - Enter detailed description of personality, habits, special needs

6. **Photos** (Step 6)
   - Upload multiple pet photos
   - Drag and drop or click to select
   - Photos improve prediction accuracy

### Viewing Predictions

After completing the form:
- Click "Ready to generate prediction!"
- View the predicted adoption timeframe
- See confidence level of the prediction
- Save the pet profile to add to your database

### Dashboard

View important statistics:
- Total pets in system
- Average prediction time
- Number of fast adopters
- Average confidence level
- Recent pet profiles with their predictions

## Data Management

### Data Fields
The system collects and stores:
- **Structured Data**: Type, breed, age, gender, color, size, fur length, health status, location, fees
- **Textual Data**: Pet name, description
- **Image Data**: Number of photos uploaded and photo URLs
- **Predictions**: Adoption timeframe, confidence level, date added
- **Metadata**: Unique pet ID, rescuer information, timestamps

### Prediction System
- **Smart Scoring Algorithm**: Considers age, breed popularity, health status, and photos
- **Dynamic Confidence Levels**: Higher confidence for pets with complete information
- **Realistic Variations**: Adds random factors to prevent identical predictions
- **Four-Tier Classification**: Fast (0-30 days), Moderate (31-90 days), Slow (100+ days)
- **Visual Indicators**: Color-coded badges and progress indicators

### CSV Export
- Click "Export CSV" button in dashboard to download all pet data
- Includes all pet information plus predictions in standardized format
- Files are compatible with Excel and Google Sheets

---

## Customization

### Adding New Breeds
Edit the `BREED_DATA` object in `data.js`:
```javascript
const BREED_DATA = {
    1: { // Dogs
        999: "New Dog Breed"
    },
    2: { // Cats  
        999: "New Cat Breed"
    }
};
```

### Adding New Colors
Edit the `COLOR_DATA` object in `data.js`:
```javascript
const COLOR_DATA = {
    99: "New Color Name"
};
```

### Integrating Your ML Model

1. **Replace the Prediction Logic**
   - Update the `predict_adoption_time` method in `backend.py`
   - Load your trained model using joblib or your preferred method
   - Ensure feature vector matches your model's expected input

2. **Update Feature Engineering**
   - Modify the `create_feature_vector` method to match your model's features
   - Update the `feature_columns` list with your model's expected features

---

## Browser Compatibility

- Chrome
- Firefox
- Safari
- Edge

---

## Data Privacy

- All data is stored locally in the browser by default
- No data is sent to external servers unless you configure the backend
- Photo uploads are processed locally
- CSV exports contain only the data you input

---

## Road Map

### ✅ Completed Features

#### Core Application
- [x] Multi-step form interface with 6 comprehensive steps
- [x] Dashboard with real-time statistics and insights
- [x] Interactive pet cards
- [x] Photo upload system with drag-and-drop
- [x] Local storage persistence for offline functionality

#### Data Management
- [x] CSV export functionality with "Export CSV" button
- [x] Pet profile creation, editing, and deletion
- [x] Prediction confidence scoring system

#### User Interface
- [x] Nice design with professional styling
- [x] Progress indicators and step navigation
- [x] View All/Show Recent toggle functionality

### Future Enhancements

#### Machine Learning Improvements
- [ ] Real-time model updating based on adoption outcomes

#### Feature Enhancements
- [ ] Advanced search and filtering capabilities
- [ ] Multi-language support for international use

#### Data Analytics
- [ ] Detailed analytics dashboard with charts and graphs
- [ ] Adoption success rate tracking over time

#### Performance Optimization
- [ ] Optimize model for different regions beyond Malaysia

---

## Contributing

To extend the application:
1. Add new features to the appropriate files
2. Update the README with new functionality
3. Test across different browsers and devices
4. Ensure responsive design is maintained

---

## License

This project is created for educational purposes. Please ensure compliance with local data protection regulations when handling pet and adopter information.
