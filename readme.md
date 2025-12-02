## Weather Data Visualizer  
Python Data Analysis & Visualization Project

### Overview  
This assignment demonstrates how Python can be used to analyze and visualize weather data. The project loads daily weather data, performs data cleaning, calculates statistical summaries using NumPy, and generates visual plots using Matplotlib. The results help in understanding temperature and humidity patterns over time.

---

### Objectives  
- Read, clean, and validate weather data stored in a CSV file  
- Perform statistical analysis on temperature and humidity  
- Aggregate data monthly to observe trends  
- Create multiple charts for data visualization  
- Save cleaned data and plots into structured folders  
- Demonstrate file handling and numerical analysis using NumPy

---

### Features  
- Convert date formats to proper datetime objects  
- Handle missing values and incorrect records  
- Daily and monthly summary statistics  
- Line plot of daily temperature  
- Bar plot of monthly average temperature  
- Scatter plot showing relationship between humidity and temperature  
- Combined visualization in a single figure  
- All outputs saved automatically in `output/` folder

---

### Project Structure  
weather-data-visualizer/

│
├── data/

│ └── DailyDelhiClimateTrain.csv
│
├── output/

│ ├── cleaned_weather.csv

│ ├── daily_temperature.png

│ ├── monthly_avg_temperature.png

│ ├── humidity_temp_scatter.png

│ └── combined_plots.png

│
└── weather_code.py\


---

### Requirements  

This project uses the following Python libraries:

pandas
numpy
matplotlib


Install dependencies if required:

```bash
pip install pandas numpy matplotlib
```
### How to Run

Place the weather CSV file inside the data folder

Open terminal in the project directory

Run the Python script:
```bash

python weather_code.py
```


View results in the output folder

### Results Generated

The script produces:

Cleaned dataset suitable for further analysis

Visual trends for temperature and humidity

Monthly insights from aggregated statistics

Combined graphical representation for interpretation

### Developed by:
Mehul Srivastava
