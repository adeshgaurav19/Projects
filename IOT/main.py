# ===================================================================
# Corrected main.py with NameError fix
# ===================================================================

import network
import urequests
import json
import utime
from machine import Pin
from dht import DHT11, InvalidChecksum

# --- Configuration ---
WIFI_SSID = "Adesh"
API_ENDPOINT = "http://adzee.pythonanywhere.com/submit_data"

# --- Hardware Initialization ---
dht_pin = Pin(28, Pin.OUT, Pin.PULL_DOWN)
sensor = DHT11(dht_pin)

# ⭐️ FIX 1: Define wlan in the global scope ⭐️
wlan = network.WLAN(network.STA_IF)

# --- Wi-Fi Connection Function ---
def connect_wifi():
    # Now this function uses the global wlan variable
    wlan.active(True)
    if not wlan.isconnected():
        print(f"Connecting to open network: {WIFI_SSID}...")
        wlan.connect(WIFI_SSID)
        max_wait = 15
        while max_wait > 0:
            if wlan.status() < 0 or wlan.status() >= 3:
                break
            max_wait -= 1
            print('.')
            utime.sleep(1)
    if wlan.isconnected():
        print('Network connected! Pico IP Address:', wlan.ifconfig()[0])
    else:
        print('Failed to connect.')


# --- Main Application Loop ---
print("Attempting to connect to WiFi...")
connect_wifi()

while True:
    try:
        utime.sleep(2)
        sensor.measure()
        temp = sensor.temperature
        hum = sensor.humidity
        
        print(f"Reading: Temp={temp}°C, Humidity={hum}%")

        payload = {"temperature": temp, "humidity": hum}
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        
        print(f"Posting data to {API_ENDPOINT}...")
        response = urequests.post(API_ENDPOINT, data=json.dumps(payload), headers=headers)

        if response.status_code == 201:
            print("Data posted successfully.")
        else:
            print(f"Failed to post data. Status: {response.status_code}, Response: {response.text}")
        response.close()

    except Exception as e:
        print(f"An error occurred: {e}")
        # ⭐️ FIX 2: This check will now work correctly ⭐️
        if not wlan.isconnected():
            print("WiFi disconnected. Attempting to reconnect...")
            connect_wifi()

    print("--------------------")
    utime.sleep(10)