import { Router } from "express";
import { FaceRecognitionController } from "./src/Controllers/FaceRecognitionController";

const routes = Router();
const faceRecognitionController = new FaceRecognitionController();

routes.use("/fr", faceRecognitionController.call);

export default routes;
