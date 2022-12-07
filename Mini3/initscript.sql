CREATE TABLE Product (
    product_category TEXT NOT NULL
    product_description TEXT NOT NULL
    price INTEGER PRIMARY KEY
    product_code TEXT NOT NULL
);
CREATE INDEX product_category ON Product(product_category);