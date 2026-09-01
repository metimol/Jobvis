from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse(request, "about.html")


@app.get("/contact")
async def contact(request: Request):
    return templates.TemplateResponse(request, "contact.html")


@app.get("/gallery")
async def gallery(request: Request):
    return templates.TemplateResponse(request, "gallery.html")


@app.get("/pricing")
async def pricing(request: Request):
    return templates.TemplateResponse(request, "pricing.html")
