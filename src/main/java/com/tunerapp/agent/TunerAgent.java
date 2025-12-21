// File: TunerAgent.java (Simplified and Corrected)
package main.java.com.tunerapp.agent;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import main.java.com.tunerapp.tuner.ParamSpace;
import io.javalin.Javalin;
import io.javalin.http.staticfiles.Location;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TunerAgent {

    public static void main(String[] args) {
        TuningService tuningService = new TuningService();
        Gson gson = new Gson();

        Javalin app = Javalin.create(config -> {
            config.staticFiles.add(staticFiles -> {
                staticFiles.directory = "/public";
                staticFiles.location = Location.CLASSPATH;
            });
        }).start(8080);

        System.out.println("Tuner Agent is running. Open http://localhost:8080");

        app.post("/api/tune", ctx -> {

            // --- THIS IS THE NEW, SIMPLIFIED JSON PARSING LOGIC ---
            ParamSpace space = new ParamSpace();
            try {
                // Gson can have trouble with mixed int/double lists. Reading all numbers as Doubles is safer.
                Type type = new TypeToken<Map<String, List<Double>>>() {}.getType();
                Map<String, List<Double>> rawMap = gson.fromJson(ctx.formParam("paramsJson"), type);

                // No casting is needed here! The flexible ParamSpace.add handles it.
                for (Map.Entry<String, List<Double>> entry : rawMap.entrySet()) {
                    space.add(entry.getKey(), entry.getValue());
                }

            } catch (Exception e) {
                ctx.status(400).json(Map.of("error", "Invalid JSON. Ensure all values in parameter lists are numbers.", "details", e.getMessage()));
                return;
            }

            String modelName = ctx.formParam("model");
            boolean isClassification = "random_forest".equals(modelName);

            String jobId = tuningService.startNewTuningJob(
                    ctx.uploadedFile("file").content(),
                    modelName,
                    space,
                    isClassification
            );

            ctx.json(Map.of("jobId", jobId));
        });

        // The status endpoint remains the same.
        app.get("/api/status/{jobId}", ctx -> {
            JobStatus status = tuningService.getJobStatus(ctx.pathParam("jobId"));
            if (status != null) {
                ctx.json(status);
            } else {
                ctx.status(404);
            }
        });
    }
}