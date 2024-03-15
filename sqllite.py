import sqlite3

## Connectt to SQlite
connection=sqlite3.connect("tourpackage.db")

# Create a cursor object to insert record,create table

cursor=connection.cursor()

## create the table
table_info="""
Create table PACKAGE(id int, AGENT_NAME VARCHAR(25),PACKAGE_TYPE VARCHAR(30),
INCLUSION VARCHAR(120),COST INT, DURATION INT);

"""
cursor.execute(table_info)

## Insert Some more records

cursor.execute('''Insert Into PACKAGE values(8, 'Adventure Expedition','adventure', "Scuba diving, kayaking, jungle trekking, zip-lining", 23000, 5)''')
cursor.execute('''Insert Into PACKAGE values(1, 'Exotic Holidays','relax', "Flights, accommodation, breakfast, city tours, beach activities", 15000, 3)''')
cursor.execute('''Insert Into PACKAGE values(2, 'Luxury Getaway','luxury', "VIP flights, 5-star accommodation, all-inclusive meals, private excursions" , 50000, 5)''')
cursor.execute('''Insert Into PACKAGE values(3, 'Family Fiesta','family', "Water park tickets, amusement park visits, family-friendly activities", 28000, 7)''')
cursor.execute('''Insert Into PACKAGE values(4, 'Adventure Expedition','adventure', "Scuba diving, kayaking, jungle trekking, zip-lining", 13000, 3)''')
cursor.execute('''Insert Into PACKAGE values(5, 'Exotic Holidays','relax', "Flights, accommodation, breakfast, city tours, beach activities", 25000, 5)''')
cursor.execute('''Insert Into PACKAGE values(6, 'Luxury Getaway','luxury', "VIP flights, 5-star accommodation, all-inclusive meals, private excursions" , 30000, 3)''')
cursor.execute('''Insert Into PACKAGE values(7, 'Family Fiesta','family', "Water park tickets, amusement park visits, family-friendly activities", 14000, 3)''')

# connection.commit()

## Disspaly ALl the records
print("The inserted records are")
# ----------------------------------------------------------------------------------------------------------------------


## create the table
table_info="""
Create table CUSTOMER(cust_id int, NAME VARCHAR(25), BOOKED_ON VARCHAR(25),  AGENT_NAME VARCHAR(25), PACKAGE_TYPE VARCHAR(30),
INCLUSION VARCHAR(120),COST INT, DURATION INT);

"""
cursor.execute(table_info)

cursor.execute('''Insert Into CUSTOMER values(4, "Jasmine", "2024-01-01", 'Adventure Expedition','adventure', "Scuba diving, kayaking, jungle trekking, zip-lining", 23000, 5)''')
cursor.execute('''Insert Into CUSTOMER values(6, 'Thompson', '2023-12-15', 'Exotic Holidays','relax', "Flights, accommodation, breakfast, city tours, beach activities", 15000, 3)''')
cursor.execute('''Insert Into CUSTOMER values(2, 'Lucas', '2024-2-13', 'Luxury Getaway','luxury', "VIP flights, 5-star accommodation, all-inclusive meals, private excursions" , 50000, 5)''')
cursor.execute('''Insert Into CUSTOMER values(3, 'Patel', '2023-12-23', 'Family Fiesta','family', "Water park tickets, amusement park visits, family-friendly activities", 28000, 7)''')
cursor.execute('''Insert Into CUSTOMER values(8, 'Ethan', '2023-10-14', 'Adventure Expedition','adventure', "Scuba diving, kayaking, jungle trekking, zip-lining", 13000, 3)''')
cursor.execute('''Insert Into CUSTOMER values(7, 'Sofia', '2024-03-12', 'Exotic Holidays','relax', "Flights, accommodation, breakfast, city tours, beach activities", 25000, 5)''')
cursor.execute('''Insert Into CUSTOMER values(1, "Johnson", "2023-11-15", 'Luxury Getaway','luxury', "VIP flights, 5-star accommodation, all-inclusive meals, private excursions" , 30000, 3)''')
cursor.execute('''Insert Into CUSTOMER values(5, "Liam", "2024-01-19", 'Family Fiesta','family', "Water park tickets, amusement park visits, family-friendly activities", 14000, 3)''')

data=cursor.execute('''Select * from PACKAGE''')

for row in data:
    print(row)

data=cursor.execute('''Select * from CUSTOMER''')

for row in data:
    print(row)

## Commit your changes int he databse
connection.commit()
connection.close()