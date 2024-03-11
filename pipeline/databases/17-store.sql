-- first trigger
DELIMETER //
CREATE TRIGGER decrease_quantity
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    UPDATE items
    SET quantity = quantity - NEW.number
    WHERE id = NEW.item_name;
END;
//
DELIMETER ;
ON items