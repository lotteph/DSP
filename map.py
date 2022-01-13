import pandas as pd
import numpy as np
import folium
import json
import matplotlib.pyplot as plt
import seaborn as sns
import predict as h
import os

IMG_HEIGHT = 300
IMG_WIDTH = 500

if not os.path.exists("images"):
    os.mkdir("images")

def get_html(name, width=IMG_WIDTH, height=IMG_HEIGHT):
    html = f'''
        <html>
            <body>
                <h1>{name}</h1><br>
                <img src="images/{name}.png" alt="{name}" width={width} height={height}>
            </body>
        </html>'''
    return html

def create_plot(data, file_name):
    sns.set_context("poster")
    plt.style.use('seaborn-poster')
    plt.plot(data['Predicted visitors'], label='Predicted visitors')
    plt.plot(data['Actual visitors'], label='Actual visitors')
    plt.xlabel('Location')
    plt.ylabel('Visitors')
    plt.legend(loc='upper left', prop={'size': 10})
    plt.savefig('images/'+file_name)
    plt.clf()

def get_predictions(data, park_name):
    predictor_cols = data.columns.to_list()[8:]
    predictions = h.predict(data, park_name, predictor_cols, 1000)
    return predictions

data = h.get_data('CSVs/resono_2020_2022.csv')

# TODO: PUT IN FUNCTION
f = open('GeoJSON files/all_parks.geojson', 'r')
geojson = json.load(f)

with open('GeoJSON files/westerpark_oost.geojson') as f:
    geo_westerparkoost = json.load(f)

with open('GeoJSON files/westerpark_centrum.geojson') as f:
    geo_westerparkcentrum = json.load(f)

with open('GeoJSON files/westerpark_gasfabriek.geojson') as f:
    geo_westerparkgasfabriek = json.load(f)

with open('GeoJSON files/westerpark_west.geojson') as f:
    geo_westerparkwest = json.load(f)

with open('GeoJSON files/erasmuspark.geojson') as f:
    geo_erasmuspark = json.load(f)

with open('GeoJSON files/oosterpark.geojson') as f:
    geo_oosterpark = json.load(f)

with open('GeoJSON files/vondelpark_west.geojson') as f:
    geo_vondelparkwest = json.load(f)

with open('GeoJSON files/vondelpark_oost3.geojson') as f:
    geo_vondelparkoost3 = json.load(f)

with open('GeoJSON files/vondelpark_oost2.geojson') as f:
    geo_vondelparkoost2 = json.load(f)

with open('GeoJSON files/vondelpark_oost1.geojson') as f:
    geo_vondelparkoost1 = json.load(f)

with open('GeoJSON files/sarphatipark.geojson') as f:
    geo_sarphatipark = json.load(f)

with open('GeoJSON files/rembrandtpark_noord.geojson') as f:
    geo_rembrandtparknoord = json.load(f)

with open('GeoJSON files/rembrandtpark_zuid.geojson') as f:
    geo_rembrandtparkzuid = json.load(f)

geos = [geo_rembrandtparknoord,
        geo_rembrandtparkzuid,
        geo_sarphatipark,
        geo_vondelparkwest,
        geo_vondelparkoost3,
        geo_vondelparkoost2,
        geo_vondelparkoost1,
        geo_oosterpark,
        geo_erasmuspark,
        geo_westerparkoost,
        geo_westerparkcentrum,
        geo_westerparkgasfabriek,
        geo_westerparkwest]

locations = ['Rembrandtpark Noord',
 'Rembrandtpark Zuid',
 'Sarphatipark',
 'Vondelpark West',
 'Vondelpark Oost 3',
 'Vondelpark Oost 2',
 'Vondelpark Oost 1',
 'Oosterpark',
 'Erasmuspark',
 'Westerpark Oost',
 'Westerpark Centrum',
 'Westergasfabriek',
 'Westerpark West']


visits = [50, 20, 40, 60, 34, 90, 120, 23, 45, 78, 34, 13, 130]

chordata = pd.DataFrame(list(zip(locations, visits)),
               columns =['Location', 'Visitors'])

style_function = lambda x: {'fillColor': '#ffffff',
                            'color':'#000000',
                            'fillOpacity': 0.0,
                            'weight': 0.5}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color':'#000000',
                                'fillOpacity': 0.5,
                                'weight': 0.5}

m = folium.Map(
    location=[52.377956, 4.897070],
    tiles="cartodbpositron",
    zoom_start=13,
)

chor = folium.Choropleth(
    geo_data=geojson,
    name="choropleth",
    data=chordata,
    columns=["Location", "Visitors"],
    key_on="properties.Naam",
    bins=6,
    fill_color="YlOrRd",
    fill_opacity=0.8,
    line_opacity=0.7,
    legend_name="Visitors",
    highlight=True,
)

chor.add_to(m)

fgp = folium.FeatureGroup(name="Parks")

for g in geos:
    name = g['features'][0]['properties']['Naam']
    gjson = folium.GeoJson(g,
                           name=name,
                           style_function=style_function,
                           highlight_function=highlight_function,
                          )
    create_plot(get_predictions(data, name), name)
    html = get_html(name)
    test = folium.Html(html, script=True)
    popup = folium.Popup(test, max_width=2650)
    gjson.add_child(popup)

    fgp.add_child(gjson)

fgp.add_to(m)

m.save('index.html')
