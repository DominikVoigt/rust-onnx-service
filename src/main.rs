use std::sync::Arc;

use axum::{
    Router,
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use ort::session::{Session, builder::GraphOptimizationLevel};
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
        .with_env_filter(LevelFilter::ERROR)
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
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

#[axum::debug_handler]
async fn handle_request(State(state): State<AppState>, mut body: Multipart) -> Response {
    let mut model_url: Option<String> = None;
    let mut model_file: Option<Vec<u8>> = None;
    let mut input: Option<Vec<f32>> = None;
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
                            return (StatusCode::BAD_REQUEST, format!("Field model_url could not be parsed to valid utf-8:\n{:?}", err)).into_response()
                        },
                    }
                },
                "model_file" => {
                    match field.bytes().await {
                        Ok(bytes) => {
                            model_file = Some(bytes.to_vec());
                         },
                        Err(err) => {
                            println!("{:?}",err);
                            return (StatusCode::BAD_REQUEST, format!("Model file could not be read:\n{:?}", err)).into_response()
                        },
                    }
                },
                "input" => {
                   match field.bytes().await {
                        Ok(input_bytes) => {
                            input = Some(match serde_json::from_slice(&input_bytes) {
                                                            Ok(json) => json,
                                                            Err(err) => return (StatusCode::BAD_REQUEST, format!("Field input could not be parsed to valid f32 vector:\n{}", err)).into_response()
                                                        });
                        },
                        Err(err) => {
                            return (StatusCode::BAD_REQUEST, format!("Field input could not be read:\n{:?}", err)).into_response()
                        },
                    }
                }
                other => {
                    return (StatusCode::BAD_REQUEST, format!("Encountered unexpected field:\n{}", other)).into_response()
                }
            },
            None => return (StatusCode::BAD_REQUEST, "Multipart does not contain a name, named mulitpart fields model_url and model_file are required").into_response()
        }
    }
    if input.is_none() {
        return (StatusCode::BAD_REQUEST, "No input was provided".to_owned()).into_response();
    }
    if model_url.is_none() {
        return (
            StatusCode::BAD_REQUEST,
            "Expected model url parameter, but was none".to_owned(),
        )
            .into_response();
    }
    let model_id = model_url.unwrap();
    let local_model;
    {
        // Check cache, and if the model is different, update
        let mut url = state.current_model_url.lock().await;
        if url.is_none() || url.as_ref().unwrap() != model_id.as_str() {
            // Load new model
            match model_file {
                Some(model_file) => {
                    let mut model = state.model.lock().await;
                    *model = match construct_model(
                        &model_file,
                        GraphOptimizationLevel::Level3,
                        NUM_THREADS,
                    ) {
                        Ok(x) => {
                            local_model = Arc::new(x);
                            Some(local_model.clone())
                        }
                        Err(err) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Encountered error creating model: {:?}", err),
                            )
                                .into_response();
                        }
                    };
                    *url = Some(model_id);
                }
                None => match url.as_ref() {
                    Some(url) => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, format!(
                            "Model URL was different from cached model ({}), but no model file was provided",
                            url
                        )).into_response();
                    }
                    None => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "No model was cached and request contained no new model".to_owned(),
                        )
                            .into_response();
                    }
                },
            }
        }
    }

    // local_model.run();
    // Once we reach here, the model id  and local model are non empty
    return StatusCode::OK.into_response();
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
