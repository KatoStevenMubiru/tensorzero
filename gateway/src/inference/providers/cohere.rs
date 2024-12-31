//! Cohere API provider implementation.
//! 
//! This module implements the Cohere chat API provider for TensorZero.
//! Supports both regular and streaming chat completions.
//!
//! # Example
//! ```rust,no_run
//! use gateway::inference::providers::{CohereProvider, InferenceProvider};
//! use gateway::model::CredentialLocation;
//!
//! let provider = CohereProvider::new(
//!     "command".to_string(),
//!     Some(CredentialLocation::Env("COHERE_API_KEY".to_string()))
//! );
//! ```

use std::env;
use async_trait::async_trait;
use futures::StreamExt;
use lazy_static::lazy_static;
use reqwest::Client;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::Instant;
use url::Url;

use crate::{
    endpoints::inference::InferenceCredentials,
    error::{Error, ErrorDetails},
    inference::types::{
        batch::BatchProviderInferenceResponse,
        ContentBlock, ModelInferenceRequest,
        ProviderInferenceResponse, ProviderInferenceResponseStream,
    },
    model::CredentialLocation,
};

use super::provider_trait::InferenceProvider;

lazy_static! {
    static ref COHERE_API_BASE: Url = {
        Url::parse("https://api.cohere.ai/v2").expect("Failed to parse COHERE_API_BASE")
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
}

#[derive(Debug, Deserialize)]
struct CohereResponse {
    text: String,
    // Add other fields based on Cohere's API response
}

#[derive(Debug, Serialize)]
struct CohereRequest<'a> {
    model: &'a str,
    messages: Vec<CohereMessage>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct CohereMessage {
    role: String,
    content: String,
}

pub struct CohereProvider {
    model_name: String,
    credentials: CredentialLocation,
}

impl CohereProvider {
    pub fn new(model_name: String, credentials: Option<CredentialLocation>) -> Self {
        Self {
            model_name,
            credentials: credentials.unwrap_or_else(|| {
                CredentialLocation::Env("COHERE_API_KEY".to_string())
            }),
        }
    }

    fn get_api_key(&self, dynamic_api_keys: &InferenceCredentials) -> Result<String, CohereError> {
        match &self.credentials {
            CredentialLocation::Env(key) => {
                env::var(key).map_err(|_| CohereError::MissingEnvironmentVariable(key.clone()))
            }
            CredentialLocation::Value(key) => Ok(key.expose_secret().to_string()),
            CredentialLocation::Dynamic(key_name) => {
                dynamic_api_keys
                    .get(key_name)
                    .ok_or_else(|| CohereError::CredentialError(
                        format!("Dynamic API key {} not found", key_name)
                    ))
                    .map(|key| key.expose_secret().to_string())
            }
        }
    }
}

#[async_trait]
impl InferenceProvider for CohereProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponse, Error> {
        let api_key = self.get_api_key(dynamic_api_keys)
            .map_err(|e| Error::new(ErrorDetails::Configuration {
                message: e.to_string(),
            }))?;

        let messages = request.messages.iter().map(|m| CohereMessage {
            role: m.role.to_string(),
            content: m.content.iter()
                .map(|block| match block {
                    ContentBlock::Text(text) => text.to_string(),
                    ContentBlock::Json(json) => json.to_string(),
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join(" "),
        }).collect();

        let cohere_request = CohereRequest {
            model: &self.model_name,
            messages,
            stream: false,
        };

        let start_time = Instant::now();
        
        // TODO: Implement actual API call and response handling
        todo!("Implement API call and response handling")
    }

    async fn infer_stream<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'a>,
        http_client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<ProviderInferenceResponseStream, Error> {
        // TODO: Implement streaming
        todo!("Implement streaming support")
    }

    async fn start_batch_inference<'a>(
        &'a self,
        requests: &'a [ModelInferenceRequest<'a>],
        client: &'a Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> impl std::future::Future<Output = Result<BatchProviderInferenceResponse, Error>> + Send + 'a {
        async move {
            // TODO: Implement batch inference
            todo!("Implement batch inference")
        }
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
            Some(CredentialLocation::Env("COHERE_API_KEY".to_string()))
        );
        assert_eq!(provider.model_name, "command");
    }

    // Add more tests...
}
