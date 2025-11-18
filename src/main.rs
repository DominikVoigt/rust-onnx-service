use axum::{Router, extract::Multipart, extract::State, routing::post};
use bytes::Buf;
use tokio::sync::Mutex;
use tract_onnx::prelude::*;

type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<Option<Arc<Model>>>>,
    current_model_url: Arc<Mutex<Option<String>>>,
}

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt::init();
    let state = AppState {
        current_model_url: Arc::new(Mutex::new(None)),
        model: Arc::new(Mutex::new(None)),
    };
    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/", post(handle_request))
        .with_state(state);

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello, World!"
}

#[axum::debug_handler]
async fn handle_request(State(state): State<AppState>, mut body: Multipart) -> Result<(), String> {
    let mut model_url: Option<String> = None;
    let mut model_file = None;
    let mut input = None;
    while let Some(field) = body.next_field().await.unwrap() {
        let name = field.name();
        match name {
            Some(name) => match name {
                "model_url" => {
                    match field.text().await {
                        Ok(url) => {
                            model_url = Some(url);
                        },
                        Err(err) => {
                            return Err(format!("Field model_url could not be parsed to valid utf-8:\n{}", err))
                        },
                    }
                },
                "model_file" => {
                    match field.bytes().await {
                        Ok(bytes) => {
                            model_file = Some(bytes)
                        },
                        Err(err) => {
                            return Err(format!("Model file could not be read:\n{}", err))
                        },
                    }
                },
                "input" => {
                   match field.text().await {
                        Ok(url) => {
                            input = Some(url)
                        },
                        Err(err) => {
                            return Err(format!("Field model_url could not be parsed to valid utf-8:\n{}", err))
                        },
                    }
                }
                other => {
                    return Err(format!("Encountered unexpected field: {}", other))
                }
            },
            None => return Err("Multipart does not contain a name, named mulitpart fields model_url and model_file are required".to_owned()),
        }
    }
    if input.is_none() {
        return Err("No input was provided".to_owned());
    }
    if model_url.is_none() {
        return Err("Expected model url parameter, but was none".to_owned());
    }
    let model_id = model_url.unwrap();
    let local_model;
    { // Check cache, and if the model is different, update
        let mut url = state.current_model_url.lock().await;
        if url.is_none() || url.as_ref().unwrap() != model_id.as_str() {
            // Load new model
            match model_file {
                Some(model_file) => {
                    let mut model = state.model.lock().await;
                    *model = match construct_model(model_file).await {
                        Ok(x) => {
                            local_model = Arc::new(x);
                            Some(local_model.clone())
                        },
                        Err(err) => {
                            return Err(format!("Encountered error creating model: {:?}", err));
                        }
                    };
                    *url = Some(model_id);
                }
                None => match url.as_ref() {
                    Some(url) => {
                        return Err(format!(
                            "Model URL was different from cached model ({}), but no model file was provided",
                            url
                        ));
                    }
                    None => {
                        return Err(
                            "No model was cached and request contained no new model".to_owned()
                        );
                    }
                },
            }
        }
    }

    local_model.run();
    // Once we reach here, the model id  and local model are non empty
    return Ok(());
}

async fn construct_model(model_file: bytes::Bytes) -> anyhow::Result<Model> {
    let mut reader = model_file.reader();
    Ok(tract_onnx::onnx()
        // load the model
        .model_for_read(&mut reader)?
        // optimize the model
        .into_optimized()?
        // make the model runnable and fix its inputs and outputs
        .into_runnable()?)
}
