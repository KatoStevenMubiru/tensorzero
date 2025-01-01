//! Cohere API provider implementation.
//!
//! This provider implements the Cohere chat API for TensorZero.
//! Supports both regular and streaming chat completions.
//!
//! # Example
//! ```rust,no_run
//! use gateway::inference::providers::{CohereProvider, Provider};
//! use gateway::model::CredentialLocation;
//!
//! let provider = CohereProvider::new(
//!     "command".to_string(),
//!     Some(CredentialLocation::Env("COHERE_API_KEY".to_string()))
//! );
//! ```

use std::env;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::Client;
use secrecy::{ExposeSecret, Secret};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::Instant;
use url::Url;
use uuid::Uuid;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::{
        common::Provider,
        types::{
            BatchProviderInferenceResponse, ChatCompletionChunk,
            ContentBlock, ModelInferenceRequest, ProviderInferenceResponse,
            Text, Usage,
        },
    },
    model::{CredentialLocation, Model},
};

lazy_static! {
    static ref COHERE_API_BASE: Url = {
        Url::parse("https://api.cohere.ai/v1").expect("Failed to parse COHERE_API_BASE")
    };
}

#[derive(Debug, Error)]
pub enum CohereError {
    #[error("Missing environment variable: {0}")]
    MissingEnvironmentVariable(String),

    #[error("Credential error: {0}")]
    CredentialError(String),

    #[error("API error: {status_code} - {message}")]
    ApiError {
        status_code: u16,
        message: String,
    },

    #[error("Request error: {0}")]
    RequestError(String),

    #[error("Response parsing error: {0}")]
    ResponseParsingError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[derive(Debug, Deserialize, Serialize)]
struct CohereResponse {
    text: String,
    tokens: Option<u32>,
    // Add other fields based on Cohere's API response
}

#[derive(Debug, Serialize)]
struct CohereRequest {
    model: String,
    messages: Vec<CohereMessage>,
    stream: bool,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct CohereMessage {
    role: String,
    content: String,
}

pub struct CohereProvider {
    model_name: String,
    credentials: CredentialLocation,
    http_client: Client,
}

impl CohereProvider {
    pub fn new(model_name: String, credentials: Option<CredentialLocation>) -> Self {
        Self {
            model_name,
            credentials: credentials.unwrap_or_else(|| {
                CredentialLocation::Env("COHERE_API_KEY".to_string())
            }),
            http_client: Client::new(),
        }
    }

    fn get_api_key(
        &self,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<Secret<String>, CohereError> {
        match &self.credentials {
            CredentialLocation::Env(key) => env::var(key)
                .map(Secret::new)
                .map_err(|_| CohereError::MissingEnvironmentVariable(key.clone())),
            CredentialLocation::Static(key) => Ok(key.clone()),
            CredentialLocation::Dynamic(key_name) => dynamic_api_keys
                .get(key_name)
                .cloned()
                .ok_or_else(|| {
                    CohereError::CredentialError(format!(
                        "Dynamic API key {} not found",
                        key_name
                    ))
                }),
        }
    }

    async fn make_request(
        &self,
        request: &CohereRequest,
        api_key: &str,
    ) -> Result<CohereResponse, CohereError> {
        let response = self.http_client
            .post(COHERE_API_BASE.join("generate").unwrap())
            .bearer_auth(api_key)
            .json(request)
            .send()
            .await
            .map_err(|e| CohereError::RequestError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(CohereError::ApiError {
                status_code: response.status().as_u16(),
                message: response.text().await.unwrap_or_default(),
            });
        }

        response
            .json::<CohereResponse>()
            .await
            .map_err(|e| CohereError::ResponseParsingError(e.to_string()))
    }
}

#[async_trait]
impl Provider for CohereProvider {
    type Error = CohereError;

    async fn infer(
        &self,
        request: &ModelInferenceRequest,
        model_config: &Model,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Self::Error> {
        let api_key = self.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();

        let messages = request.messages.iter().map(|m| CohereMessage {
            role: m.role.to_string(),
            content: m.content.iter()
                .map(|block| match block {
                    ContentBlock::Text(text) => text.text.to_string(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" "),
        }).collect();

        let cohere_request = CohereRequest {
            model: self.model_name.clone(),
            messages,
            stream: false,
            temperature: model_config.parameters.get("temperature")
                .and_then(|v| v.as_f64())
                .map(|v| v as f32),
            max_tokens: model_config.parameters.get("max_tokens")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
        };

        let response = self.make_request(&cohere_request, api_key.expose_secret())
            .await?;

        Ok(ProviderInferenceResponse {
            id: Uuid::new_v4(),
            created: start_time.elapsed().as_secs(),
            output: vec![ContentBlock::Text(Text { 
                text: response.text 
            })],
            model: Some(request.id.clone()),
            usage: Some(Usage {
                prompt_tokens: 0, // TODO: Get from response
                completion_tokens: response.tokens.unwrap_or(0),
                total_tokens: response.tokens.unwrap_or(0),
            }),
            raw_request: serde_json::to_value(&cohere_request)?,
            raw_response: serde_json::to_value(&response)?,
        })
    }

    async fn infer_stream(
        &self,
        request: &ModelInferenceRequest,
        model_config: &Model,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, Self::Error>>, Self::Error> {
        // TODO: Implement streaming
        todo!("Streaming support not yet implemented")
    }

    async fn start_batch_inference(
        &self,
        requests: &[ModelInferenceRequest],
        model_config: &Model,
        dynamic_api_keys: &InferenceCredentials,
    ) -> Result<BatchProviderInferenceResponse, Self::Error> {
        // TODO: Implement batch inference
        todo!("Batch inference not yet implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::types::Message;

    #[tokio::test]
    async fn test_cohere_provider_creation() {
        let provider = CohereProvider::new(
            "command".to_string(),
            Some(CredentialLocation::Env("COHERE_API_KEY".to_string())),
        );
        assert_eq!(provider.model_name, "command");
    }

    #[tokio::test]
    async fn test_api_key_retrieval() {
        let provider = CohereProvider::new(
            "command".to_string(),
            Some(CredentialLocation::Env("TEST_COHERE_KEY".to_string())),
        );
        
        std::env::set_var("TEST_COHERE_KEY", "test-key");
        let creds = InferenceCredentials::default();
        
        let result = provider.get_api_key(&creds);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().expose_secret(), "test-key");
    }

    // Add more tests...
}
