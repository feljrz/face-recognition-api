import { Request, Response } from "express";

class FaceRecognitionController {
  async call(request: Request, response: Response) {
    const { PythonShell } = require("python-shell");

    const options = {
      mode: "text",
      pythonPath: "python3",
      scriptPath: "/home/felipe/Documents/Alianca/image-explorer/my_tests/",
    };

    return await PythonShell.run(
      "interface.py",
      options,
      function (err: any, results: any) {
        if (err) console.log(err);
        return response.json(results);
      }
    );
  }
}
export { FaceRecognitionController };
