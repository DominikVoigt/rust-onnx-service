use std::{fs::read, sync::Arc};

use axum::{
    Form, Router, extract::{DefaultBodyLimit, Multipart, State}, http::StatusCode, response::{IntoResponse, Response}, routing::post
};
use ort::{session::{Session, builder::GraphOptimizationLevel}, value::Tensor};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tracing_subscriber::filter::LevelFilter;
type Model = Session;

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<Option<Arc<Model>>>>,
    current_model_url: Arc<Mutex<Option<String>>>,
}

static NUM_THREADS: usize = 4;

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt()
        //.with_env_filter(LevelFilter::ERROR)
        .finish();
    let state = AppState {
        current_model_url: Arc::new(Mutex::new(None)),
        model: Arc::new(Mutex::new(None)),
    };
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
    model_url: String
}

#[axum::debug_handler]
async fn handle_request(State(state): State<AppState>, Form(request): Form<PredictionRequest>) -> Response {
    println!("Received Request");
    let model_file_name = "random_forest_heating_2h_short-term.onnx";
    let model_file = read(format!("./resources/test_files/{}", model_file_name)).unwrap();
    let mut model = construct_model(&model_file, GraphOptimizationLevel::Level3, 1).unwrap();

    let input: Vec<f32> = serde_json::from_str(&request.input).unwrap();
    let input = Tensor::from_array(([1usize, 12], input)).unwrap();
    let res = model.run(ort::inputs![input]).unwrap();
    let res = res["variable"].try_extract_array::<f32>().unwrap()[[0, 0]];
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
