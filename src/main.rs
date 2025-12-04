use std::{fs::read, sync::Arc};

use axum::{
    Form, Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use moka::future::Cache;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing_subscriber::EnvFilter;
type Model = Session;

#[derive(Clone)]
struct AppState {
    // As we can only run a single inference through each model, we need to mutex it
    model_cache: Cache<String, Arc<Mutex<Session>>>,
}

// MAX_CACHED_MODELS needs to be > 0
static MAX_CACHED_MODELS: u64 = 4;

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt().finish();
    let model_cache: Cache<String, Arc<Mutex<Session>>> = Cache::new(MAX_CACHED_MODELS);

    let state = AppState { model_cache };
    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/", post(handle_request))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 100))
        .with_state(state);

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind(":::3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[derive(Serialize, Deserialize)]
struct PredictionRequest {
    input: String,
    model_url: String,
}

#[axum::debug_handler]
async fn handle_request(
    State(state): State<AppState>,
    Form(request): Form<PredictionRequest>,
) -> Response {
    println!("Received Request");
    let model_url = request.model_url.as_str();
    let model = state.model_cache.get(model_url).await;
    let local_model = match model {
        Some(model) => {
            // Acquire shared pointer to cached model (stays valid even if model is moved out of the cache during the request)
            model.clone()
        }
        None => {
            // Model is not cached, thus download model, and store in cache
            let client = reqwest::Client::new();
            let res = client.get(model_url).send().await;
            let res = match res {
                Ok(response) => match response.bytes().await {
                    Ok(model_file) => {
                        construct_model(&model_file, GraphOptimizationLevel::Level3, 1)
                    }
                    Err(err) => Err(err.into()),
                },
                Err(err) => Err(err.into()),
            };
            match res {
                Ok(model) => {
                    let model = Arc::new(Mutex::new(model));
                    state
                        .model_cache
                        .insert(model_url.to_owned(), model.clone())
                        .await;
                    model
                }
                Err(err) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, format!("{:?}", err))
                        .into_response();
                }
            }
        }
    };

    let input: Vec<f32> = match serde_json::from_str(&request.input) {
        Ok(input) => input,
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response(),
    };
    let input = match Tensor::from_array(([1usize, 12], input)) {
        Ok(input) => input,
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response(),
    };
    let mut model_lock = local_model.lock().await;
    let res = match model_lock.run(ort::inputs![input]) {
        Ok(out) => out,
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response(),
    };
    let res = match res["variable"].try_extract_array::<f32>() {
        Ok(input) => input[[0, 0]],
        Err(err) => return (StatusCode::BAD_REQUEST, format!("{:?}", err)).into_response(),
    };
    return (StatusCode::OK, res.to_string()).into_response();
}

fn construct_model(
    model_bytes: &[u8],
    level: GraphOptimizationLevel,
    threads: usize,
) -> anyhow::Result<Model> {
    // let mut reader = model_file.reader();
    let model = Session::builder()?
        .with_optimization_level(level)?
        .with_intra_threads(threads)?
        .commit_from_memory(model_bytes)?;
    Ok(model)
}

#[cfg(test)]
mod test {
    use std::{fs::read, time::Instant};

    use ort::{session::builder::GraphOptimizationLevel, value::Tensor};

    use crate::construct_model;

    #[test]
    fn test_model_creation_d() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        //File::open(path)
        let level = GraphOptimizationLevel::Disable;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_creation_l1() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        //File::open(path)
        let level = GraphOptimizationLevel::Level1;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_creation_l2() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        //File::open(path)
        let level = GraphOptimizationLevel::Level2;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_creation_l3() {
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        //File::open(path)
        let level = GraphOptimizationLevel::Level3;
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let start = Instant::now();
        let level_str = format!("{:?}", level);
        let model = construct_model(&model_file, level, 4).unwrap();
        let elapsed = start.elapsed();
        println!(
            "Loading the model with level {} took: {}",
            level_str,
            elapsed.as_secs_f64()
        )
    }

    #[test]
    fn test_model_inference() {
        let data = vec![
            0.820045_f32,
            0.820045,
            0.69701624,
            0.69701624,
            0.6086147,
            0.6086147,
            0.51213145,
            0.44877553,
            0.44877553,
            0.44877553,
            0.33946013,
            0.2616396,
        ];
        let input = Tensor::from_array(([1usize, 12], data)).unwrap();
        let model_file_name = "random_forest_heating_2h_short-term.onnx";
        let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
        let mut model = construct_model(&model_file, GraphOptimizationLevel::Level3, 1).unwrap();
        let start = Instant::now();
        let res = model.run(ort::inputs![input]).unwrap();
        let elapsed = start.elapsed().as_micros();
        println!("Elapsed time: {}", elapsed);
        let res = res["variable"].try_extract_array::<f32>().unwrap()[[0, 0]];
        println!("Res: {:?}", res)
    }
}
