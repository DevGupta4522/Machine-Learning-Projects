/*******************************************************/
/* SQL for Data Analysis - Furniture Rental Store       */
/*******************************************************/

/* Before running these queries, create the furniture_rental DB using
   furniture_rental_schema.sql and load furniture_rental_data.sql */

USE furniture_rental;

/* Email campaign for customers of Store 2:
   First name, last name, and email of customers from Store 2 */
SELECT first_name, last_name, email
FROM customer
WHERE store_id = 2;

/* How many furniture items rent for $19.99? */
SELECT COUNT(*) FROM furniture_item
WHERE rental_rate = 19.99;

/* Rental rate distribution: how many items fall into each rental rate */
SELECT rental_rate, COUNT(*) AS total_number_of_items
FROM furniture_item
GROUP BY rental_rate;

SELECT rental_rate, COUNT(*) AS total_number_of_items
FROM furniture_item
GROUP BY 1;

/* Which condition rating do we have the most items in? */
SELECT condition_rating, COUNT(*) AS total_number_of_items
FROM furniture_item
GROUP BY 1;

/* Which condition rating is most prevalent in each store's inventory? */
SELECT s.store_id, f.condition_rating, COUNT(f.condition_rating) AS total_number_of_items
FROM store s
JOIN inventory i ON s.store_id = i.store_id
JOIN furniture_item f ON f.item_id = i.item_id
GROUP BY 1,2;

/* Mailing list for an upcoming promotion */
SELECT c.customer_id, c.first_name, c.last_name, a.address
FROM customer c
JOIN address a ON c.address_id = a.address_id;

/* List of furniture items by name, category, and material */
SELECT f.item_name, c.name, m.name
FROM furniture_item f
JOIN furniture_category fc ON fc.item_id = f.item_id
JOIN category c ON fc.category_id = c.category_id
JOIN material m ON m.material_id = f.material_id;

/* How many times has each item been rented out? */
SELECT i.item_id, f.item_name, COUNT(i.item_id) AS total_number_of_rental_times
FROM rental r
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN furniture_item f ON f.item_id = i.item_id
GROUP BY i.item_id
ORDER BY 3 DESC;

/* Revenue per furniture item */
SELECT i.item_id, f.item_name, COUNT(i.item_id) AS total_number_of_rental_times, f.rental_rate, COUNT(i.item_id)*f.rental_rate AS revenue_per_item
FROM rental r
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN furniture_item f ON f.item_id = i.item_id
GROUP BY i.item_id
ORDER BY 5 DESC;

/* Highest-spending customers, to target for rewards */
SELECT c.customer_id, SUM(p.amount) AS "Total Spending"
FROM customer c
JOIN payment p ON c.customer_id = p.customer_id
GROUP BY 1
ORDER BY 2 DESC;

/* Which store has historically brought in the most revenue? */
SELECT s.store_id, SUM(p.amount) AS "Total Spending"
FROM store s
JOIN inventory i ON i.store_id = s.store_id
JOIN rental r ON r.inventory_id = i.inventory_id
JOIN payment p ON p.rental_id = r.rental_id
GROUP BY 1
ORDER BY 2 DESC;

/* Rentals per month */
SELECT LEFT(rental_date,7) AS "Month", COUNT(*)
FROM rental
GROUP BY 1;

/* Rentals per calendar month (Jan => X, Feb => Y, etc.) */
SELECT date_format(rental_date,"%M") AS "Month", COUNT(*)
FROM rental
GROUP BY 1
ORDER BY 2 DESC;

/* Date of the very first rental */
SELECT MIN(rental_date)
FROM rental;

/* Date of the most recent rental */
SELECT MAX(rental_date)
FROM rental;

/* For each item, when was it first and last rented out? */
SELECT f.item_name AS "Item Name", MIN(r.rental_date) AS "First Rented Date", MAX(r.rental_date) AS "Last Rented Date"
FROM furniture_item f
JOIN inventory i ON f.item_id = i.item_id
JOIN rental r ON r.inventory_id = i.inventory_id
GROUP BY 1;

/* Last rental date of every customer */
SELECT c.customer_id, c.first_name, c.last_name, MAX(r.rental_date) AS "Last Rental Date"
FROM customer c
JOIN rental r ON r.customer_id = c.customer_id
GROUP BY 1;

/* Revenue per month */
SELECT LEFT(payment_date,7) AS "Month", SUM(amount) AS "Revenue Per Month"
FROM payment
GROUP BY 1;

/* Distinct renters per month */
SELECT LEFT(rental_date,7) AS "Month",
	COUNT(DISTINCT(rental_id)) AS "Total Rentals",
	COUNT(DISTINCT(customer_id)) AS "Number Of Unique Renters",
    COUNT(DISTINCT(rental_id))/COUNT(DISTINCT(customer_id)) AS "Average Rentals Per Renter"
FROM rental
GROUP BY 1;

/* Number of distinct items rented each month */
SELECT i.item_id, f.item_name, LEFT(r.rental_date,7) AS "Month", COUNT(i.item_id) AS "Total Number Of Rental Times"
FROM rental r
JOIN inventory i ON r.inventory_id = i.inventory_id
JOIN furniture_item f ON f.item_id = i.item_id
GROUP BY i.item_id, LEFT(r.rental_date,7)
ORDER BY 1, 2, 3;

/* Number of rentals in Sofas, Chairs, and Beds */
SELECT c.name, COUNT(c.name) AS "Number of Rentals"
FROM furniture_item f
JOIN furniture_category fc ON fc.item_id = f.item_id
JOIN category c ON c.category_id = fc.category_id
JOIN inventory i ON f.item_id = i.item_id
JOIN rental r ON r.inventory_id = i.inventory_id
WHERE c.name IN ("Sofas", "Chairs", "Beds")
GROUP BY 1;

/* Customers who have rented at least 3 times */
SELECT c.customer_id, CONCAT(c.first_name, " ", c.last_name) AS "Customer Name", COUNT(c.customer_id) AS "Total Rentals"
FROM customer c
JOIN rental r ON c.customer_id = r.customer_id
GROUP BY 1
HAVING COUNT(c.customer_id) >= 3
ORDER BY 1;

/* How much revenue has one single store made from "Fair" and "Poor" condition items */
SELECT s.store_id, f.condition_rating, SUM(p.amount) AS "Total Revenue"
FROM store s
JOIN inventory i ON i.store_id = s.store_id
JOIN rental r ON r.inventory_id = i.inventory_id
JOIN payment p ON p.rental_id = r.rental_id
JOIN furniture_item f ON f.item_id = i.item_id
WHERE f.condition_rating IN ("Fair", "Poor")
GROUP BY 1,2;

/*******************************************************/

/* Active customers (active = 1) */
DROP TEMPORARY TABLE IF EXISTS tbl_active_customers;
CREATE TEMPORARY TABLE tbl_active_customers(
SELECT c.*, a.phone
FROM customer c
JOIN address a ON a.address_id = c.address_id
WHERE c.active = 1);

/* Reward customers: who has rented at least 5 times */
DROP TEMPORARY TABLE IF EXISTS tbl_rewards_customer;
CREATE TEMPORARY TABLE tbl_rewards_customer(
SELECT r.customer_id, COUNT(r.customer_id) AS total_rents, max(r.rental_date) AS last_rental_date
FROM rental r
GROUP BY 1
HAVING COUNT(r.customer_id) >= 5);

/* Reward customers who are also active */
SELECT au.customer_id, au.first_name, au.last_name, au.email
FROM tbl_rewards_customer ru
JOIN tbl_active_customers au ON au.customer_id = ru.customer_id;

/* All reward customers with phone number */
SELECT ru.customer_id, c.email, au.phone
FROM tbl_rewards_customer ru
LEFT JOIN tbl_active_customers au ON au.customer_id = ru.customer_id
JOIN customer c ON c.customer_id = ru.customer_id;
