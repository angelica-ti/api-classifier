from fastapi import APIRouter, UploadFile
from app.model.preprocessing import Preprocessing
from app.model.classifier import Classifier


router = APIRouter(
    prefix="/classify-image",
    tags=["classify-image"],
    responses={404: {"description": "Not found"}},
)

prep = Preprocessing()
clf = Classifier()

@router.post("/classify-image")
async def post_classify(file: UploadFile):
    image = file.file.read()
    image_processed = prep.apply(image)
    clf.load_model()
    prediction = clf.predict(image_processed)    
    return {"prediction":prediction}