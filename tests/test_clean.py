import pandas as pd

from skyrecover.data.clean import clean, hhmm_to_minutes


def raw_rows(**overrides):
    base = {
        "FlightDate": "2024-01-05", "Reporting_Airline": "DL", "Tail_Number": "N1",
        "Flight_Number_Reporting_Airline": 100, "Origin": "ATL", "Dest": "LAX",
        "CRSDepTime": 1000, "CRSArrTime": 1150, "CRSElapsedTime": 290,
        "DepDelay": 0, "ArrDelay": 0, "Cancelled": 0, "Diverted": 0, "Distance": 1946,
        "CarrierDelay": 0, "WeatherDelay": 0, "NASDelay": 0, "SecurityDelay": 0,
        "LateAircraftDelay": 0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_hhmm_to_minutes():
    assert hhmm_to_minutes(pd.Series([1430, 5, 2400])).tolist() == [870, 5, 1440]


def test_arrival_uses_local_time_at_destination():
    # The time-zone bug: ATL 10:00 Eastern + 4h50 lands 11:50 PACIFIC, not 14:50.
    d = clean(raw_rows())
    assert d["sched_arr"].iloc[0] == pd.Timestamp("2024-01-05 11:50")


def test_red_eye_lands_next_day():
    d = clean(raw_rows(Origin="LAX", Dest="ATL", CRSDepTime=2330, CRSArrTime=650, CRSElapsedTime=260))
    assert d["sched_arr"].iloc[0] == pd.Timestamp("2024-01-06 06:50")


def test_filters_other_airlines_and_diverted():
    raw = pd.concat([raw_rows(), raw_rows(Reporting_Airline="AA"), raw_rows(Diverted=1)])
    assert len(clean(raw, airline="DL")) == 1


def test_drops_missing_tail_numbers():
    assert len(clean(raw_rows(Tail_Number=None))) == 0
