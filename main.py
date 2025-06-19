from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from  models import ItemPayload

from get_text_from_pdf import get_text_from_pdf
import os
import shutil
import PyPDF2
import redis
import json

from sentence_transformer import identify_document_type


app = FastAPI()

redis_client = redis.StrictRedis(host="127.0.0.1", port=6379, db=0, decode_responses=True)

@app.get("/")
def root():
    return {"message": "Hello World"}

# Route to add an item
@app.post("/items/{item_name}/{quantity}")
def add_item(item_name: str, quantity: int) -> dict[str, ItemPayload]:
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0.")

    # Check if item already exists
    item_id_str: str | None = redis_client.hget("item_name_to_id", item_name)

    if item_id_str is not None:
        item_id = int(item_id_str)
        redis_client.hincrby(f"item_id:{item_id}", "quantity", quantity)
    else:
        # Generate an ID for the item
        item_id: int = redis_client.incr("item_ids")
        redis_client.hset(
            f"item_id:{item_id}",
            mapping={
                "item_id": item_id,
                "item_name": item_name,
                "quantity": quantity,
            },
        )
        # Create a set so we can search by name too
        redis_client.hset("item_name_to_id", item_name, item_id)

    return {
        "item": ItemPayload(item_id=item_id, item_name=item_name, quantity=quantity)
    }


# Route to list a specific item by ID but using Redis
@app.get("/items/{item_id}")
def list_item(item_id: int) -> dict[str, dict[str, str]]:
    if not redis_client.hexists(f"item_id:{item_id}", "item_id"):
        raise HTTPException(status_code=404, detail="Item not found.")
    else:
        return {"item": redis_client.hgetall(f"item_id:{item_id}")}


@app.get("/items")
def list_items() -> dict[str, list[ItemPayload]]:
    items: list[ItemPayload] = []
    stored_items: dict[str, str] = redis_client.hgetall("item_name_to_id")

    for name, id_str in stored_items.items():
        item_id: int = int(id_str)

        item_name_str: str | None = redis_client.hget(f"item_id:{item_id}", "item_name")
        if item_name_str is not None:
            item_name: str = item_name_str
        else:
            continue  # skip this item if it has no name

        item_quantity_str: str | None = redis_client.hget(
            f"item_id:{item_id}", "quantity"
        )
        if item_quantity_str is not None:
            item_quantity: int = int(item_quantity_str)
        else:
            item_quantity = 0

        items.append(
            ItemPayload(item_id=item_id, item_name=item_name, quantity=item_quantity)
        )

    return {"items": items}


# Route to delete a specific item by ID but using Redis
@app.delete("/items/{item_id}")
def delete_item(item_id: int) -> dict[str, str]:
    if not redis_client.hexists(f"item_id:{item_id}", "item_id"):
        raise HTTPException(status_code=404, detail="Item not found.")
    else:
        item_name: str | None = redis_client.hget(f"item_id:{item_id}", "item_name")
        redis_client.hdel("item_name_to_id", f"{item_name}")
        redis_client.delete(f"item_id:{item_id}")
        return {"result": "Item deleted."}


# Route to remove some quantity of a specific item by ID but using Redis
@app.delete("/items/{item_id}/{quantity}")
def remove_quantity(item_id: int, quantity: int) -> dict[str, str]:
    if not redis_client.hexists(f"item_id:{item_id}", "item_id"):
        raise HTTPException(status_code=404, detail="Item not found.")

    item_quantity: str | None = redis_client.hget(f"item_id:{item_id}", "quantity")

    # if quantity to be removed is higher or equal to item's quantity, delete the item
    if item_quantity is None:
        existing_quantity: int = 0
    else:
        existing_quantity: int = int(item_quantity)
    if existing_quantity <= quantity:
        item_name: str | None = redis_client.hget(f"item_id:{item_id}", "item_name")
        redis_client.hdel("item_name_to_id", f"{item_name}")
        redis_client.delete(f"item_id:{item_id}")
        return {"result": "Item deleted."}
    else:
        redis_client.hincrby(f"item_id:{item_id}", "quantity", -quantity)
        return {"result": f"{quantity} items removed."}




UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Redis client
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "File must be a PDF"})
    file_location: str = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Extract text from the saved PDF
    text = ""
    with open(file_location, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    id_result: tuple[str, str] = identify_document_type(text)
    result: dict[str, Any] = {"filename": file.filename, "text": text, "document_type": id_result[0], "confidence_score": id_result[1]}
    
    # Save the return object in Redis as JSON
    redis_client.set(f"pdf_result:{file.filename}", json.dumps(result))
    
    return result


@app.get("/pdfs/")
def get_all_pdfs() -> dict[str, list[dict[str, Any]]]:
    # Find all keys that match the pattern for stored PDFs
    keys = redis_client.keys("pdf_result:*")
    pdfs = []
    for key in keys:
        value = redis_client.get(key)
        if value:
            pdfs.append(json.loads(value))
    return {"pdfs": pdfs}


@app.get("/pdfs/{filename}")
def get_pdf(filename: str):
    key = f"pdf_result:{filename}"
    value = redis_client.get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="PDF not found")
    return json.loads(value)
