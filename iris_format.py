from pydantic import BaseModel


class Iris_Fromat(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float

    