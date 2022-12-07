from flask import Flask,render_template,request, redirect, url_for
import requests
import pandas as pd
import sqlite3



with sqlite3.connect("productdb.db",check_same_thread=False) as con:
    con.execute("CREATE TABLE IF NOT EXISTS Product (product_category TEXT NOT NULL, product_description TEXT NOT NULL, price INTEGER PRIMARY KEY, product_code TEXT NOT NULL);")
    con.commit() # Creating table 




app = Flask(__name__, template_folder = 'templates')
def connectToDb():
    with sqlite3.connect("productdb.db") as con:
        return sqlite3.connect("productdb.db") #connection to database created 



@app.route('/')
def home():
    return render_template('index.html') #homepage

@app.route('/product', methods=["GET","POST"])
def product():
  
    with connectToDb() as con: # connection to database so user can enter data 
        if request.method=="POST":
            con.execute("INSERT INTO Product(product_category,product_description,price,product_code) values(:product_category,:product_description,:price,:product_code)", request.form)
            
    df = pd.read_sql("select * from Product", con) # output, reading the data on page
    
    print(df)
    return render_template('productdata.html', data=df.to_records()) #recorded data 
    

@app.route('/retrievedata', methods=["GET","POST"])
def retrievedata():
    return render_template('retrievedata.html') #User input to retrieve category
 
@app.route('/data', methods =["POST","GET"])
def data():
    product_category = request.form.get("product_category") 
    product_description = request.form.get("product_description")
    price = request.form.get("price")
    product_code = request.form.get("product_code") #requests to get data entered

    with sqlite3.connect("productdb.db") as con:
        if (product_category != ""):
            data = pd.read_sql("SELECT * from Product WHERE product_category=?", con, params=(product_category,)) #Selecting specific category
        else:
            data = pd.read_sql("SELECT * FROM Product",con)  #Printing all if no input
            print(data)

    return render_template('data.html', data = data )


app.run(debug=True, port =8080)

# DNS: https://andreeacontis.dynv6.net/
