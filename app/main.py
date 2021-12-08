import json
import pickle

import pandas as pd
import numpy as np
import io
import random
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.cluster import AgglomerativeClustering
from flask import (
    Flask,
    request,
    render_template,
    session,
    redirect,
    render_template_string,
    Response,
    url_for,
    make_response,
    current_app,
    send_file,
)
from sklearn.tree import DecisionTreeClassifier
from fpdf import FPDF
import config
import joblib

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

data = pd.read_csv("spotify_dataset.csv")
data_view = pd.read_csv("spotify_dataset.csv")
# cars_view["PriceinGermany"] = cars_view["PriceinGermany"].replace(np.NAN, "no info")
# cars_view.drop(columns=["PriceinUK"], inplace=True)
# data.columns = [
#     "name",
#     "subtitle",
#     "acceleration_in_sec",
#     "top_speed_km_per_h",
#     "range_km",
#     "efficiency_wh_per_hour",
#     "fast_charge_speed_km_per_h",
#     "drive",
#     "number_of_seats",
#     "price_in_euros",
#     "price_to_drop",
# ]


data.drop(columns=["Weeks Charted", "Song ID", "Week of Highest Charting"], inplace=True)
data_view.drop(columns=["Weeks Charted", "Song ID", "Week of Highest Charting"], inplace=True)

data['Streams'] = data['Streams'].str.replace(",", "").astype("int")
data['Artist Followers'] = data['Artist Followers'].str.replace(" ", "1").astype("int")

# numeric_columns = data.loc[
#                   :,
#                   [
#                       "acceleration_in_sec",
#                       "top_speed_km_per_h",
#                       "range_km",
#                       "efficiency_wh_per_hour",
#                       "fast_charge_speed_km_per_h",
#                       "number_of_seats",
#                       "price_in_euros",
#                   ],
#                   ]
#
# data.acceleration_in_sec = data.acceleration_in_sec.str.replace("sec", "").astype(
#     "float"
# )
# data.top_speed_km_per_h = data.top_speed_km_per_h.str.replace("km/h", "").astype("int")
# data.range_km = data.range_km.str.replace("km", "").astype("int")
# data.efficiency_wh_per_hour = data.efficiency_wh_per_hour.str.replace(
#     "Wh/km", ""
# ).astype("int")
# data.fast_charge_speed_km_per_h = data.fast_charge_speed_km_per_h.str.replace(
#     "km/h", ""
# )
# data.price_in_euros = (
#     data.price_in_euros.str.replace("€", "").str.replace(",", ".").astype("float")
# )
# data.loc[data.price_in_euros.isnull(), "price_in_euros"] = round(
#     data.price_in_euros.mean(), 2
# )
info = data.describe(percentiles=[])

app = Flask(
    __name__, static_folder=config.STATIC_FOLDER, template_folder=config.TEMPLATE_FOLDER
)

matplotlib.pyplot.switch_backend('Agg')


@app.route("/", methods=("POST", "GET"))
def html_table():
    return render_template(
        "index.html",
        tables=[data_view.to_html(classes="data")],
        titles=data_view.columns.values,
    )


@app.route("/steams_count", methods=("POST", "GET"))
def html_steams_count():
    print(data.tail())
    listening = steams_count_find()
    create_pdf("Отчёт по количеству прослушиваний", f"Общее количество прослушиваний составляет {listening} раз",
               "static/steams_count.png",
               "steams_count")
    return render_template(
        "steams_count.html",
        listening=listening
    )


@app.route("/steams_count.pdf")
def send_pdf_streams_count():
    return send_file(config.STATIC_FOLDER + "/steams_count.pdf")


@app.route("/average_followers", methods=("POST", "GET"))
def html_average_followers():
    average = average_followers_find()
    create_pdf("Отчёт по среднему количству подписчиков",
               f"Среднее количество подписчиков у исполнителей составляет {average} человек.",
               image_path="static/average_followers.png", save_name="average_followers")
    return render_template(
        "average_followers.html",
        average=average
    )


@app.route("/average_followers.pdf")
def send_pdf_max():
    return send_file(config.STATIC_FOLDER + "/average_followers.pdf")


@app.route("/classification", methods=("POST", "GET"))
def html_clustering():
    print(data.tail())
    names, values, score = classification()
    print(names)
    print(list(values))
    clusters = pd.DataFrame(data=[values], columns=names, index=["Важность"])
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'font/DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)
    pdf.write(txt="Классификация", h=15)
    pdf.ln(25)
    pdf.write(txt=clusters.to_string(col_space=21), h=7)
    pdf.ln(10)
    pdf.write(
        txt="Исходя из классификации, можно сказать, что наибольшее влияние на эффективность работы электродвигателя "
            "оказывает дальность расстояния, а именно, сколько километров сможет проехать электроавтомобиль на одном "
            "заряде. Качество модели:" + str(
            score), h=7)
    pdf.output("static/classification.pdf")
    return render_template(
        "classification.html",
        score=round(score, 5),
        tables=[clusters.to_html(classes="data")]
    )


@app.route("/classification.pdf")
def send_pdf_classification():
    return send_file(config.STATIC_FOLDER + "/classification.pdf")


@app.route("/cross_validation", methods=("POST", "GET"))
def cross_validation():  # Кросс-валидация качество модели
    return render_template("cross_validation.html", validation=str(cross_validation_start()))


# --Поиск общего количества прослушиваний--
def steams_count_find():
    sum = 0
    for i in data['Streams']:
        sum += i

    fig, ax = plt.subplots()
    ax.set_title('Количество прослушиваний')
    plt.style.context('bmh')
    ax.set_xlabel('Исполнитель')
    ax.set_ylabel('Количество прослушиваний (10 млн)')
    plt.plot(data['Streams'])
    plt.grid('on')
    plt.savefig('static/steams_count.png')
    return sum


# --Поиск среднего количества подписчиков--
def average_followers_find():
    sum = 0
    res = 0
    for i in range(len(data["Artist Followers"])):
        sum += int(data["Artist Followers"][i])
    res = round(sum / len(data["Artist Followers"]), 0)

    fig, ax = plt.subplots()
    ax.set_title('Распределение подписчиков исполнителей')
    plt.style.context('bmh')
    ax.set_xlabel('Количество подписчиков (10 млн)')
    ax.set_ylabel('Количество исполнителей')
    plt.hist(data["Artist Followers"], bins=15)
    plt.grid('on')
    plt.savefig('static/average_followers.png')
    return res


def classification():
    x = data[['top_speed_km_per_h', 'range_km', "number_of_seats"]]
    y = data['efficiency_wh_per_hour']
    clf = DecisionTreeClassifier()
    clf.fit(x, y)
    print(clf.feature_names_in_)
    # Вывод важности
    print(clf.feature_importances_)
    print(clf.score(x, y))  # верность классификации
    names = ["Макс скорость", "Макс дистанция", "Количество сидений"]
    save_file(clf, "static/model.pkl")
    return names, clf.feature_importances_, clf.score(x, y),


def create_pdf(title, text, image_path, save_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'font/DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', 14)
    pdf.write(txt=title, h=15)
    pdf.ln(20)
    pdf.image(image_path, x=10, y=20, w=170)
    pdf.ln(130)  # ниже на 85
    pdf.write(
        txt=text,
        h=7)
    pdf.output("static/" + save_name + ".pdf")


def cross_validation_start():
    x = data[['top_speed_km_per_h', 'range_km', "number_of_seats"]]
    y = data['efficiency_wh_per_hour']

    clf = load_file("static/model.pkl")

    cv_results = cross_val_score(clf, x, y, cv=3)

    return cv_results.mean()


def load_file(filename):
    return joblib.load(filename)


def save_file(model, filename):
    joblib.dump(model, filename)
