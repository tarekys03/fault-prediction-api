import sqlite3

conn = sqlite3.connect("obd_data.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS obd_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Timestamp TEXT DEFAULT CURRENT_TIMESTAMP,

    Engine_RPM INTEGER,
    Coolant_Temp_C INTEGER,
    Oil_Temp_C INTEGER,
    Idle_Status BOOLEAN,
    Engine_Load_Percent INTEGER,
    Ignition_Timing_Deg REAL,
    MAP_kPa INTEGER,
    MAF_gps REAL,
    Battery_Voltage_V REAL,
    Charging_System_Status TEXT,

    O2_Sensor_V REAL,
    Catalytic_Converter_Percent INTEGER,
    EGR_Status TEXT,
    Vehicle_Speed_kmh INTEGER,

    Transmission_Gear TEXT,
    Brake_Status TEXT,
    Tire_Pressure_psi REAL,
    Ambient_Temp_C INTEGER,
    Battery_Age_Months INTEGER,
    Fuel_Level_Percent INTEGER,

    Predicted_Fault TEXT,
    Prediction_Message TEXT
)
""")

conn.commit()
conn.close()
print("✅ Done creating the SQLite database and table.")


# table_name = "obd_data"  
# cur.execute(f"DELETE FROM {table_name}")

# conn.commit()

# conn.close()

# print("✅ Done deleting all data from the SQLite database table.")
