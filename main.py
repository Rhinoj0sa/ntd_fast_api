from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from  models import ItemPayload
app = FastAPI()

grocery_list: dict[int, ItemPayload] = {}


@app.get("/")
def root():
    return {"message": "Hello World"}

# Route to add a item
@app.post("/items/{item_name}/{quantity}")
def add_item(item_name: str, quantity: int):
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be greater than 0.")
    # if item already exists, we'll just add the quantity.
    # get all item names
    items_ids = {item.item_name: item.item_id if item.item_id is not None else 0 for item in grocery_list.values()}
    if item_name in items_ids.keys():
        # get index of item_name in item_ids, which is the item_id
        item_id = items_ids[item_name]
        grocery_list[item_id].quantity += quantity
# otherwise, create a new item
    else:
        # generate an ID for the item based on the highest ID in the grocery_list
        item_id = max(grocery_list.keys()) + 1 if grocery_list else 0
        grocery_list[item_id] = ItemPayload(
            item_id=item_id, item_name=item_name, quantity=quantity
        )

    return {"item": grocery_list[item_id]}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "File must be a PDF"})
    contents = await file.read()
    # You can process the PDF here
    return JSONResponse(content={"filename": file.filename, "size": len(contents)})
