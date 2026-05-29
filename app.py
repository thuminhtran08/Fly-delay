import numpy as np
from flask import Flask, request, jsonify, render_template
from model import processInput
import pandas as pd
import math
import datetime
import re
from difflib import get_close_matches

app = Flask(__name__)

def string_to_time(time_string):
    if pd.isnull(time_string):
        return np.nan
    if time_string == 2400:
        time_string = 0
    time_string = "{0:04d}".format(int(time_string))
    return datetime.time(int(time_string[0:2]), int(time_string[2:4]))

def func(x):
    return x.hour * 3600 + x.minute * 60 + x.second

def normalize(s):
    return re.sub(r'\s+', ' ', str(s).replace('\xa0', ' ')).strip().lower()

def normalize_no_paren(s):
    s = re.sub(r'\(.*?\)', '', str(s))
    return re.sub(r'\s+', ' ', s.replace('\xa0', ' ')).strip().lower()

def safe_int(value, field_name):
    if value is None or value == "":
        raise ValueError("Thieu du lieu: " + field_name)
    try:
        return int(value)
    except Exception:
        raise ValueError("Du lieu khong hop le: " + field_name)

def find_airport(airports_df, query):
    q = query.strip()
    if re.fullmatch(r'[A-Za-z]{3}', q):
        row = airports_df[airports_df['IATA_CODE'] == q.upper()]
        if not row.empty:
            return q.upper(), row.iloc[0]
    q_norm    = normalize(q)
    q_norm_np = normalize_no_paren(q)
    all_names      = airports_df['AIRPORT'].tolist()
    all_names_norm = [normalize(n)          for n in all_names]
    all_names_np   = [normalize_no_paren(n) for n in all_names]
    if q_norm in all_names_norm:
        idx = all_names_norm.index(q_norm)
        return airports_df.iloc[idx]['IATA_CODE'], airports_df.iloc[idx]
    if q_norm_np in all_names_np:
        idx = all_names_np.index(q_norm_np)
        return airports_df.iloc[idx]['IATA_CODE'], airports_df.iloc[idx]
    for i, (nn, nnp) in enumerate(zip(all_names_norm, all_names_np)):
        if q_norm_np and (q_norm_np in nnp or nnp in q_norm_np):
            return airports_df.iloc[i]['IATA_CODE'], airports_df.iloc[i]
    close = get_close_matches(q_norm_np, all_names_np, n=1, cutoff=0.75)
    if close:
        idx = all_names_np.index(close[0])
        return airports_df.iloc[idx]['IATA_CODE'], airports_df.iloc[idx]
    return None, None

def render_error(msg):
    return """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Loi</title>
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600&display=swap" rel="stylesheet">
    <style>body{font-family:'Sora',sans-serif;background:#f8fafc;display:flex;
    align-items:center;justify-content:center;min-height:100vh;margin:0}
    .box{background:#fff;border-radius:16px;padding:40px 48px;
    box-shadow:0 4px 24px rgba(0,0,0,0.08);max-width:520px;text-align:center}
    h2{color:#dc2626}p{color:#64748b;line-height:1.6}
    a{display:inline-block;margin-top:24px;padding:12px 32px;
    background:linear-gradient(135deg,#38bdf8,#818cf8);color:#fff;
    border-radius:10px;text-decoration:none;font-weight:600}</style></head>
    <body><div class="box"><h2>&#9888;&#65039; Khong the xu ly yeu cau</h2>
    <p>""" + msg + """</p><a href="/index">&#8592; Thu lai</a></div></body></html>"""

@app.route('/')
def home():
    return render_template('model.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        day_of_week  = safe_int(request.form.get('dayOfWeek'), 'dayOfWeek')
        sd_val       = safe_int(request.form.get('sd'),        'sd')
        sa_val       = safe_int(request.form.get('sa'),        'sa')
        ddelay_val   = safe_int(request.form.get('ddelay'),    'ddelay')
        airline_full = request.form.get('airline',     '').strip()
        origin_input = request.form.get('origin',      '').strip()
        dest_input   = request.form.get('destination', '').strip()

        if not airline_full: return render_error("Vui long chon hang hang khong.")
        if not origin_input: return render_error("Vui long chon san bay xuat phat.")
        if not dest_input:   return render_error("Vui long chon san bay den.")

        airports = pd.read_csv('airports.csv')
        origin_code, origin_row = find_airport(airports, origin_input)
        if origin_code is None:
            return render_error("Khong tim thay san bay xuat phat: " + origin_input)
        dest_code, dest_row = find_airport(airports, dest_input)
        if dest_code is None:
            return render_error("Khong tim thay san bay den: " + dest_input)

        carrier_code = airline_full.split(',')[0].strip()
        input_ = {
            "dayOfWeek": day_of_week, "carrier": carrier_code,
            "sd": sd_val, "sa": sa_val, "ddelay": ddelay_val, "origin": origin_code,
        }

        errors = pd.read_csv('errors/errors.csv')
        carrier_errors = errors[errors['airline'].str.strip() == carrier_code]
        if carrier_errors.empty:
            return render_error("Khong tim thay hang: " + carrier_code)
        error_ = float(carrier_errors.iloc[0]['error'])

        org_lat   = float(origin_row['LATITUDE'])
        org_lng   = float(origin_row['LONGITUDE'])
        dest_lat  = float(dest_row['LATITUDE'])
        dest_lng  = float(dest_row['LONGITUDE'])

        flights_distance = 3963.0 * math.acos(max(-1.0, min(1.0,
            math.sin(math.radians(org_lat))  * math.sin(math.radians(dest_lat)) +
            math.cos(math.radians(org_lat))  * math.cos(math.radians(dest_lat)) *
            math.cos(math.radians(dest_lng  - org_lng)))))
        input_["dist"] = flights_distance

        from model import safe_predict
        df, model = processInput(input_)
        if df is None or model is None:
            return render_error("Mo hinh du bao bi loi.")
        prediction = safe_predict(model, df)

        d_time = sd_val + ddelay_val
        if d_time < 0:           d_time = 2359 + d_time
        if d_time % 100 > 59:    d_time += 40
        if d_time > 2359:        d_time -= 2400
        mer_d = "am" if 0 <= d_time < 1200 else "pm"
        arr_d = "am" if sa_val < 1200 else "pm"

        arr_delay = round(prediction)
        res_time  = sa_val + arr_delay
        if res_time < 0:         res_time = 2359 + res_time
        if res_time % 100 > 59:  res_time += 40
        if res_time > 2359:      res_time -= 2400

        travel_time = func(string_to_time(res_time)) - func(string_to_time(d_time))
        if travel_time < 0:      travel_time += 86400

        # Truyền toàn bộ data qua tojson – tránh Jinja2 parse float âm
        flask_data = {
            "org_lat":     org_lat,
            "org_lng":     org_lng,
            "dest_lat":    dest_lat,
            "dest_lng":    dest_lng,
            "origin":      str(origin_row['AIRPORT']),
            "origin_code": origin_code,
            "destination": str(dest_row['AIRPORT']),
            "dest_code":   dest_code,
            "prediction":  round(prediction, 2),
            "error":       round(error_, 2),
            "d_time":      d_time,
            "sa":          sa_val,
            "mer_d":       mer_d,
            "arr_d":       arr_d,
            "distance":    round(flights_distance, 2),
            "t_hrs":       int(travel_time // 3600),
            "t_mins":      int((travel_time % 3600) // 60),
            "carrierN":    carrier_code,
        }

        return render_template("result.html", flask_data=flask_data)

    except Exception as e:
        import traceback; traceback.print_exc()
        return render_error("Loi he thong: " + str(e))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        df, model = processInput(data)
        from model import safe_predict
        result = safe_predict(model, df)
        return jsonify({"predicted_delay_minutes": round(result, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)