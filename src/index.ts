#!/usr/bin/env node
/**
 * MCP Server for Vision Tools API
 */
// Load environment variables from .env file
import dotenv from 'dotenv';
dotenv.config();
import fs from "fs";
import os from "os";
import path from "path";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ReadResourceResultSchema,
  ImageContent,
  ResourceContentsSchema,
  TextContent,
  type Tool,
  type CallToolResult,
  type CallToolRequest,
  Resource,
  ReadResourceResult
} from "@modelcontextprotocol/sdk/types.js";
import { z, ZodError } from 'zod';
import { jsonSchemaToZod } from 'json-schema-to-zod';
import axios, { type AxiosRequestConfig, type AxiosError } from 'axios';
import FormData from 'form-data';
import sharp from 'sharp';
import { fileURLToPath } from 'url';

/**
 * Server configuration
 */
export const SERVER_NAME = "vision-tools-api";
export const SERVER_VERSION = "0.1.0";
export const API_BASE_URL = "https://api.va.landing.ai";
/**
 * MCP Server instance
 */
const server = new Server({ name: SERVER_NAME, version: SERVER_VERSION }, { capabilities: { tools: {} } });
/**
 * Map of tool definitions by name
 */
const toolDefinitionMap: Map<string, McpToolDefinition> = new Map([
    ["wsi_embedding", {
            name: "wsi_embedding",
            description: `Wsi Embedding`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" }, "mpp": { "anyOf": [{ "type": "number", "minimum": 0 }, { "type": "null" }], "title": "Mpp", "description": "Microns per pixel" } }, "type": "object", "required": ["image"], "title": "WSIEmbeddingRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/wsi-embedding",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["qr_reader", {
            name: "qr_reader",
            description: `Qr Reader`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" } }, "type": "object", "required": ["image"], "title": "QRReaderRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/qr-reader",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["owlv2", {
            name: "owlv2",
            description: `Owlv2`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" }, "prompts": { "items": { "type": "string" }, "type": "array", "title": "Prompts" }, "confidence": { "anyOf": [{ "type": "number", "maximum": 1, "minimum": 0 }, { "type": "null" }], "title": "Confidence", "default": 0.2 } }, "type": "object", "required": ["image", "prompts"], "title": "Owlv2Request", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/owlv2",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["depth_anything_v2", {
            name: "depth_anything_v2",
            description: `Depth Anything V2`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" }, "grayscale": { "anyOf": [{ "type": "boolean" }, { "type": "null" }], "title": "Grayscale", "default": false } }, "type": "object", "required": ["image"], "title": "DepthAnythingV2Request", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/depth-anything-v2",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["loca", {
            name: "loca",
            description: `Loca`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" }, "bbox": { "anyOf": [{ "items": { "anyOf": [{ "type": "integer" }, { "type": "number" }] }, "type": "array", "maxItems": 4, "minItems": 4 }, { "type": "null" }], "title": "Bbox" } }, "type": "object", "required": ["image"], "title": "LocaRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/loca",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["florencev2", {
            name: "florencev2",
            description: `Florencev2`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "anyOf": [{ "type": "string", "maxLength": 50000000 }, { "type": "null" }], "title": "Image" }, "images": { "anyOf": [{ "items": { "type": "string" }, "type": "array", "maxItems": 50000000 }, { "type": "null" }], "title": "Images" }, "video": { "anyOf": [{ "type": "string", "maxLength": 50000000 }, { "type": "null" }], "title": "Video" }, "video_bytes": { "anyOf": [{ "type": "string", "format": "binary" }, { "type": "null" }], "title": "Video Bytes" }, "task": { "type": "string", "enum": ["<CAPTION>", "<CAPTION_TO_PHRASE_GROUNDING>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>", "<DENSE_REGION_CAPTION>", "<OPEN_VOCABULARY_DETECTION>", "<OD>", "<OCR>", "<OCR_WITH_REGION>", "<REGION_PROPOSAL>", "<REFERRING_EXPRESSION_SEGMENTATION>", "<REGION_TO_SEGMENTATION>", "<REGION_TO_CATEGORY>", "<REGION_TO_DESCRIPTION>"], "title": "PromptTask" }, "prompt": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Prompt" } }, "type": "object", "required": ["task"], "title": "FlorenceV2Request", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/florence2",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["nsfw_classification", {
            name: "nsfw_classification",
            description: `Nsfw Classification`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" } }, "type": "object", "required": ["image"], "title": "NSFWRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/nsfw-classification",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["florencev2_qa", {
            name: "florencev2_qa",
            description: `Florencev2 Qa`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" }, "question": { "type": "string", "title": "Question" } }, "type": "object", "required": ["image", "question"], "title": "FlorenceQARequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/florence2-qa",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["pose_detection", {
            name: "pose_detection",
            description: `Pose Detection`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" } }, "type": "object", "required": ["image"], "title": "PoseDetectionRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/pose-detector",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["barcode_reader", {
            name: "barcode_reader",
            description: `Barcode Reader`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" } }, "type": "object", "required": ["image"], "title": "BarcodeReaderRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/barcode-reader",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["internlm_xcomposer2", {
            name: "internlm_xcomposer2",
            description: `Internlm Xcomposer2`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: multipart/form-data)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/internlm-xcomposer2",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "multipart/form-data",
            securityRequirements: []
        }],
    ["video_temporal_localization", {
            name: "video_temporal_localization",
            description: `Performs temporal localization on a video using the specified model

Args:
    model: model used
    prompt (str): prompt for the task
    video: video file to perform temporal localization on
    chunk_length (float, optional): length of each chunk in seconds
    chunk_length_seconds (float, optional): alternative len for chunk in seconds
    chunk_length_frames (int, optional): length of each chunk in frames
    baseten_inference_sender: The dependency for the baseten inference sender

Returns:
    VideoTemporalLocalizationResponse | JSONResponse`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/video-temporal-localization",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["florence2_sam2", {
            name: "florence2_sam2",
            description: `Florence2 Sam2`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/florence2-sam2",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["text_to_object_detection", {
            name: "text_to_object_detection",
            description: `Text To Od`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/text-to-object-detection",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["visual_prompts_to_object_detection", {
            name: "visual_prompts_to_object_detection",
            description: `Visual Prompts To Od`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: multipart/form-data)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/visual-prompts-to-object-detection",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "multipart/form-data",
            securityRequirements: []
        }],
    ["countgd", {
            name: "countgd",
            description: `Countgd`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "properties": { "image": { "type": "string", "maxLength": 50000000, "title": "Image" }, "prompt": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Prompt" }, "visual_prompts": { "anyOf": [{ "items": { "items": { "type": "number" }, "type": "array" }, "type": "array" }, { "type": "null" }], "title": "Visual Prompts" } }, "type": "object", "required": ["image"], "title": "CountGDRequest", "description": "The JSON request body." } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/countgd",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/json",
            securityRequirements: []
        }],
    ["flux1", {
            name: "flux1",
            description: `Performs Image Generation or Mask Inpainting using the specified task. Can generate an image from a text prompt or inpaint a mask on an image. If the user asks for image generation, use this.

Args:
    data (Dict[str, Any]): The input dictionary containing the following keys:
        - task (Flux1Task): The task to perform using the model:
            either image generation ("generation")
            or mask inpainting ("inpainting").
        - prompt (str): The text prompt describing the desired modifications.
        - image (Image.Image): The original image to be modified.
        - mask_image (Image.Image): The mask image indicating areas to be inpainted.
        - height (\`int\`, *optional*):
            The height in pixels of the generated image.
            This is set to 512 by default.
        - width (\`int\`, *optional*):
            The width in pixels of the generated image.
            This is set to 512 by default.
        - num_inference_steps (\`int\`, *optional*, defaults to 28):
        - guidance_scale (\`float\`, *optional*, defaults to 3.5):
            Guidance scale as defined in Classifier-Free Diffusion Guidance.
            Higher guidance scale encourages to generate images
            that are closely linked to the text \`prompt\`,
            usually at the expense of lower image quality.
        - num_images_per_prompt (\`int\`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        - max_sequence_length (\`int\` defaults to 512):
            Maximum sequence length to use with the \`prompt\`.
            to make generation deterministic.
        - strength (\`float\`, *optional*, defaults to 0.6):
            Indicates extent to transform the reference \`image\`.
            Must be between 0 and 1.
            A value of 1 essentially ignores \`image\`.
        - seed (\`int\`, *optional*): The seed to use for the random number generator.
            If not provided, a random seed is used.
Returns:
    Flux1Response | JSONResponse`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/flux1",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["image_to_text", {
            name: "image_to_text",
            description: `Image To Text`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/image-to-text",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["text_to_instance_segmentation", {
            name: "text_to_instance_segmentation",
            description: `Text To Seg`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/text-to-instance-segmentation",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["classification", {
            name: "classification",
            description: `Performs Image Generation or Mask Inpainting using the specified model.

Args:
    data (Dict[str, Any]): The input dictionary containing the following keys:
        - model (ClassificationModel):
            The model to be used for image classification.
            Currently, only SigLIP Zero-shot Image Classification is supported.
        - image (Image.Image):
            The original image to be classified.
        - labels (List[str]):
            The candidate labels for the image classification model.

    baseten_inference_sender: Dependency Injection for
        sending request to the baseten server.

Returns:
    ClassificationResponse | JSONResponse
        The list of classification results, each containing a label and a score.`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/classification",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["sam2", {
            name: "sam2",
            description: `Sam2`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/sam2",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["document_analysis", {
            name: "document_analysis",
            description: `Performs Document Analysis using the specified parameters

Args:
    data (DocAnalysisRequest): The input data containing the following fields:
        - image (UploadFile | None): The image file to analyze.
        - pdf (UploadFile | None): The PDF file to analyze.
        - parse_text (bool): Whether to parse and analyze text.
        - parse_tables (bool): Whether to parse and analyze tables.
        - parse_figures (bool): Whether to parse and analyze figures.
        - summary_verbosity (Verbosity): Verbosity of the AI summary for each chunk.
        - return_chunk_crops (bool): return the crop of each chunk as base64 image.
        - return_page_crops (bool): return the crop of each page as base64 image.
        - caption_format (OutputFormat): Format of the caption for each chunk.
        - response_format (OutputFormat): Format of the overall HTTP response.
        - filename (str | None): Optional filename to be integrated into API output.`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/document-analysis",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["agentic_document_analysis", {
            name: "agentic_document_analysis",
            description: `Performs Document Analysis using the specified parameters

Args:
    data (DocAnalysisRequest): The input data containing the following fields:
        - image (UploadFile | None): The image file to analyze.
        - pdf (UploadFile | None): The PDF file to analyze.
        - include_marginalia (bool): Whether to include marginalia (header, footer, etc.) on the response.
        - include_metadata_in_markdown (bool): include chunk metadata in the markdown output as HTML comments (invisible to markdown renderers)
    llm (str): The LLM provider to use, extracted from header.`,
            inputSchema: { "type": "object", "properties": { "pages": { "anyOf": [{ "type": "string" }, { "type": "null" }], "description": "Which pages to process, separated by comma and starting from 0. For example, to process the first 3 pages, use '0,1,2'. ", "title": "Pages" }, "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "llm": { "anyOf": [{ "enum": ["openai", "azure-openai", "anthropic"], "type": "string" }, { "type": "null" }], "default": "openai", "title": "Llm" }, "AGDERateLimitTier": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "title": "Agderatelimittier" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/agentic-document-analysis",
            executionParameters: [{ "name": "pages", "in": "query" }, { "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }, { "name": "llm", "in": "header" }, { "name": "AGDERateLimitTier", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["embeddings", {
            name: "embeddings",
            description: `Embeddings`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/embeddings",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["custom_object_detection", {
            name: "custom_object_detection",
            description: `Object Detection`,
            inputSchema: { "type": "object", "properties": { "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/custom-object-detection",
            executionParameters: [],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["agentic_object_detection", {
            name: "agentic_object_detection",
            description: `Agentic Od`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/agentic-object-detection",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["docling", {
            name: "docling",
            description: `Docling`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "authorization": { "anyOf": [{ "type": "string" }, { "type": "null" }], "title": "Authorization" }, "requestBody": { "type": "string", "description": "Request body (content type: application/x-www-form-urlencoded)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/docling",
            executionParameters: [{ "name": "timeout", "in": "query" }, { "name": "authorization", "in": "header" }],
            requestBodyContentType: "application/x-www-form-urlencoded",
            securityRequirements: []
        }],
    ["license_plate", {
            name: "license_plate",
            description: `Performs license plate recognition

Args:
    video (UploadFile): The video file to analyze.`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "requestBody": { "type": "string", "description": "Request body (content type: multipart/form-data)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/license-plate",
            executionParameters: [{ "name": "timeout", "in": "query" }],
            requestBodyContentType: "multipart/form-data",
            securityRequirements: []
        }],
    ["activity_recognition", {
            name: "activity_recognition",
            description: `Performs activity recognition

Args:
    video (UploadFile): The video file to analyze.
    prompt (str): The prompt to guide activity recognition.
    specificity (Specificity, optional): Detail level in the response (low, medium, high, max).
    with_audio (bool, optional): Whether to process audio as well. Defaults to False.`,
            inputSchema: { "type": "object", "properties": { "timeout": { "anyOf": [{ "type": "integer" }, { "type": "null" }], "default": 480, "title": "Timeout" }, "requestBody": { "type": "string", "description": "Request body (content type: multipart/form-data)" } }, "required": ["requestBody"] },
            method: "post",
            pathTemplate: "/v1/tools/activity-recognition",
            executionParameters: [{ "name": "timeout", "in": "query" }],
            requestBodyContentType: "multipart/form-data",
            securityRequirements: []
        }],
]);
/**
 * Security schemes from the OpenAPI spec
 */
const securitySchemes = {};
server.setRequestHandler(ListToolsRequestSchema, async () => {
    const toolsForClient: Tool[] = Array.from(toolDefinitionMap.values()).map(def => ({
        name: def.name,
        description: "Note: Any files passed to image, pdf, or video parameters must be absolute paths or uris, no relative paths. Here is what this tool does: " + def.description,
        inputSchema: def.inputSchema
    }));
    return { tools: toolsForClient };
});
server.setRequestHandler(CallToolRequestSchema, async (request: CallToolRequest): Promise<any> => {
    const { name: toolName, arguments: toolArgs } = request.params;
    const toolDefinition = toolDefinitionMap.get(toolName);
    if (!toolDefinition) {
        console.error(`Error: Unknown tool requested: ${toolName}`);
        return { content: [{ type: "text", text: `Error: Unknown tool requested: ${toolName}` }] };
    }
    return await executeApiTool(toolName, toolDefinition, toolArgs ?? {}, securitySchemes);
});


function saveBase64Image(jsonString: string, filePath: string): boolean {
    try {
        const parsedObject = JSON.parse(jsonString);
        const base64Data: string = parsedObject.data[0];
        if (!isValidBase64(base64Data)) {
            return false;
        }
        
        const imageBuffer: Buffer = Buffer.from(base64Data, 'base64');
        if (filePath.includes('/dev/null')) {
            return true;
        }
        fs.writeFileSync(filePath, imageBuffer);
        return true;
    } catch (error: unknown) {
        console.error("Error saving base64 image:", error);
        return false;
    }
}


function isValidBase64(str: string): boolean {
    const base64Regex: RegExp = /^[A-Za-z0-9+/]*={0,2}$/;
    if (!base64Regex.test(str)) {
        return false;
    }
    
    if (str.length % 4 !== 0 || str.length < 1000) {
        return false;
    }
    
    try {
        const decoded: Buffer = Buffer.from(str, 'base64');
        const reEncoded: string = decoded.toString('base64');
        const normalizedInput: string = str.replace(/=+$/, '');
        const normalizedOutput: string = reEncoded.replace(/=+$/, '');
        
        return normalizedInput === normalizedOutput;
    } catch (e) {
        return false;
    }
}


interface LoadFileOptions {
    fileType?: string;
    contentType?: string;
    skipImageProcessing?: boolean;
    sharpOptions?: {
        formatOptions?: Record<string, unknown>;
    };
    keepAlpha?: boolean;
    outputFormat?: string;
    filename?: string;
}

interface LoadedFile {
    buffer: Buffer;
    contentType: string;
    filename: string;
    originalSize: number;
}

/**
 * Loads a file from a URL or local path and returns it as a buffer with metadata
 * @param {string} source - The URL or local file path
 * @param {Object} options - Additional options for processing
 * @returns {Object} - An object with buffer, contentType, and filename
 */
async function loadFile(source: string, options: LoadFileOptions = {}): Promise<LoadedFile> {
    try {
        let buffer: Buffer;
        let contentType: string;
        let filename: string = source.split('/').pop()?.split('\\').pop() || 'file';
        
        // Handle remote URL vs local file path
        if (source.startsWith('http://') || source.startsWith('https://')) {
            const response = await axios.get(source, { 
                responseType: 'arraybuffer',
            });
            buffer = Buffer.from(response.data, 'binary');
            contentType = response.headers['content-type'] || '';
        } else {
            buffer = await fs.promises.readFile(source);
            
            // Determine content type from file extension
            const ext: string = source.split('.').pop()?.toLowerCase() || '';
            const mimeTypes: Record<string, string> = {
                png: 'image/png',
                jpg: 'image/jpeg',
                jpeg: 'image/jpeg',
                gif: 'image/gif',
                webp: 'image/webp',
                bmp: 'image/bmp',
                svg: 'image/svg+xml',
                mp4: 'video/mp4',
                mov: 'video/quicktime',
                avi: 'video/x-msvideo',
                webm: 'video/webm',
                pdf: 'application/pdf'
            };
            contentType = mimeTypes[ext] || 'application/octet-stream';
        }
        
        // Override content type if specified in options
        if (options.contentType) {
            contentType = options.contentType;
        }
        
        // Process based on content type
        if (contentType.startsWith('image/') && !options.skipImageProcessing) {
            // Use sharp with configurable options
            const sharpOptions = options.sharpOptions || {};
            let sharpInstance = sharp(buffer);
            
            // Apply removeAlpha only if not explicitly disabled
            if (options.keepAlpha !== true) {
                sharpInstance = sharpInstance.removeAlpha();
            }
            
            // Apply format conversion if specified, default to PNG
            const outputFormat: string = options.outputFormat || 'png';
            sharpInstance = sharpInstance.toFormat(outputFormat as keyof sharp.FormatEnum, sharpOptions.formatOptions || {});
            
            // Process the image
            buffer = await sharpInstance.toBuffer();
            contentType = `image/${outputFormat}`;
            
            // Update filename extension if needed
            if (filename.includes('.')) {
                filename = filename.substring(0, filename.lastIndexOf('.')) + '.' + outputFormat;
            } else {
                filename = `${filename}.${outputFormat}`;
            }
        } 
        else if (contentType.startsWith('video/')) {
            // Ensure filename has extension
            if (!filename.includes('.')) {
                const ext: string = contentType.split('/')[1] || 'mp4';
                filename = `${filename}.${ext}`;
            }
        } 
        else if (contentType === 'application/pdf') {
            // Ensure filename has extension
            if (!filename.includes('.')) {
                filename = `${filename}.pdf`;
            }
        }
        
        // Allow custom filename override
        if (options.filename) {
            filename = options.filename;
        }
        
        return {
            buffer,
            contentType,
            filename,
            originalSize: buffer.length,
        };
    } catch (error: unknown) {
        console.error('Error loading or processing file:', (error as Error).message);
        throw error;
    }
}



async function fileToBase64(input: string, options: LoadFileOptions = {}): Promise<string> {
  try {
    let buffer;
    const fileType = options.fileType || detectFileType(input);
    
    // Get file buffer from URL or local path
    if (input.startsWith('http://') || input.startsWith('https://')) {
      const response = await axios.get(input, { responseType: 'arraybuffer' });
      buffer = Buffer.from(response.data, 'binary');
    } else {
      buffer = await fs.promises.readFile(input);
    }
    
    switch (fileType) {
      case 'image':
        const processedImage = await sharp(buffer).removeAlpha().toFormat('png').toBuffer();
        return processedImage.toString('base64');
      
      default:
        return buffer.toString('base64');
    }
  } catch (err) {
    if (err instanceof Error) {
        throw new Error(`Failed to process file: ${err.message}`);
    } else {
        throw new Error('Failed to process file: Unknown error');
    }
  }
}

// Helper function to detect file type based on extension or URL
function detectFileType(input: string): FileType {
    const lowerInput: string = input.toLowerCase();
    
    if (lowerInput.endsWith('.pdf')) {
        return 'pdf';
    } else if (/\.(mp4|mov|avi|wmv|flv|mkv|webm)$/.test(lowerInput)) {
        return 'video';
    } else if (/\.(jpg|jpeg|png|gif|bmp|webp|svg|tiff)$/.test(lowerInput)) {
        return 'image';
    } else {
        return 'binary';
    }
}

type FileType = 'pdf' | 'video' | 'image' | 'binary';

/**
 * Resizes a base64 image to 512x512 and returns the new base64 string.
 * @param base64Image - A base64 encoded image string (e.g., data:image/png;base64,...)
 * @returns A Promise that resolves to a resized base64 string
 */
export async function resizeBase64Image(base64Image: string): Promise<string> {
  const imageBuffer = Buffer.from(base64Image, 'base64');

  // Resize using sharp
  const resizedBuffer = await sharp(imageBuffer)
    .resize(512, 512)
    .toBuffer();

  return resizedBuffer.toString('base64');
}

/**
 * Executes an API tool with the provided arguments
 *
 * @param toolName Name of the tool to execute
 * @param definition Tool definition
 * @param toolArgs Arguments provided by the user
 * @param allSecuritySchemes Security schemes from the OpenAPI spec
 * @returns Call tool result
 */
interface ToolArgs {
    [key: string]: unknown;
    authorization?: string;
    requestBody?: string | Record<string, unknown>;
}

interface McpToolDefinition {
    name: string;
    description: string;
    inputSchema: any;
    method: string;
    pathTemplate: string;
    executionParameters: { name: string, in: string }[];
    requestBodyContentType?: string;
    securityRequirements: any[];
}

type JsonObject = Record<string, any>;

async function executeApiTool(
    toolName: string,
    definition: McpToolDefinition,
    toolArgs: JsonObject,
    allSecuritySchemes: Record<string, any>
) {
    const apiKey = process.env.VISION_AGENT_API_KEY;
    let outputDir = process.env.OUTPUT_DIRECTORY;
    const image_display_enabled = process.env.IMAGE_DISPLAY_ENABLED === 'true';
    toolArgs = { ...toolArgs, authorization: `Basic ${apiKey}` };
    if (typeof outputDir === 'string') {
        if (outputDir.startsWith('~')) {
            outputDir = path.join(os.homedir(), outputDir.slice(1));
        }
        if (outputDir.startsWith('.')) {
            const __filename = fileURLToPath(import.meta.url);
            const __dirname = path.dirname(__filename);
            outputDir = path.join(__dirname, outputDir);
        }
        fs.mkdirSync(outputDir, { recursive: true });
        console.error(outputDir);
    }
    try {
        // Validate arguments against the input schema
        let validatedArgs: ToolArgs;
        try {
            const zodSchema = getZodSchemaFromJsonSchema(definition.inputSchema, toolName);
            const argsToParse = (typeof toolArgs === 'object' && toolArgs !== null) ? toolArgs : {};
            validatedArgs = zodSchema.parse(argsToParse);
        } catch (error) {
            if (error instanceof ZodError) {
                const validationErrorMessage = `Invalid arguments for tool '${toolName}': ${error.errors.map(e => `${e.path.join('.')} (${e.code}): ${e.message}`).join(', ')}`;
                return { content: [{ type: 'text', text: validationErrorMessage }] };
            } else {
                const errorMessage = error instanceof Error ? error.message : String(error);
                return { content: [{ type: 'text', text: `Internal error during validation setup: ${errorMessage}` }] };
            }
        }

        // Prepare URL, query parameters, headers, and request body
        let urlPath = definition.pathTemplate;
        const queryParams: Record<string, unknown> = {};
        const headers: Record<string, string> = { 'Accept': 'application/json' };
        let requestBodyData: unknown = undefined;

        // Apply parameters to the URL path, query, or headers
        definition.executionParameters.forEach((param) => {
            const value = validatedArgs[param.name];
            if (typeof value !== 'undefined' && value !== null) {
                if (param.in === 'path') {
                    urlPath = urlPath.replace(`{${param.name}}`, encodeURIComponent(String(value)));
                } else if (param.in === 'query') {
                    queryParams[param.name] = value;
                } else if (param.in === 'header') {
                    headers[param.name.toLowerCase()] = String(value);
                }
            }
        });

        // Ensure all path parameters are resolved
        if (urlPath.includes('{')) {
            throw new Error(`Failed to resolve path parameters: ${urlPath}`);
        }

        // Construct the full URL
        const requestUrl = API_BASE_URL ? `${API_BASE_URL}${urlPath}` : urlPath;

        // Handle request body if needed
        const form = new FormData();
        if (definition.requestBodyContentType && typeof validatedArgs['requestBody'] === 'string') {
            requestBodyData = validatedArgs['requestBody'];
            if (typeof requestBodyData === 'string') {
                const match = requestBodyData.match(/[?&]?(image|pdf|video)=([^&\s]+)/);
                if (match) {
                    const type = match[1];
                    let url = match[2];
                    if (url.startsWith('@')) {
                        url = url.slice(1);
                    }
                    const normalizedPath = path.normalize(url);
                    if (!path.isAbsolute(normalizedPath)) {
                        throw new Error("Please provide a global (absolute) file path instead of a local one.");
                    }
                    const loadedFile = await loadFile(url, { fileType: type });
                    form.append(type, loadedFile.buffer, { filename: loadedFile.filename, contentType: loadedFile.contentType });
                }
            }
        }

        let config: AxiosRequestConfig = {};
        if (typeof requestBodyData === 'string') {
            const parsed = Object.fromEntries(new URLSearchParams(requestBodyData));
            Object.entries(parsed).forEach(([key, value]) => {
                if (key === 'image' || key === 'pdf' || key === 'video') {
                    return;
                }
                form.append(key, value);
            });

            // Prepare the axios request configuration
            config = {
                method: definition.method.toUpperCase(),
                url: requestUrl,
                params: queryParams,
                headers: headers,
                data: form,
            };
        } else {
            requestBodyData = validatedArgs['requestBody'];
            const fileTypes = ['image', 'video', 'pdf'];
            
            // Type guard to check if an object is a valid record
            const isValidRecord = (data: unknown): data is Record<string, unknown> => {
                return typeof data === 'object' && data !== null;
            };

            for (const fileType of fileTypes) {
                if (isValidRecord(requestBodyData) && 
                    fileType in requestBodyData && 
                    requestBodyData[fileType]) {
                        let req_string = requestBodyData[fileType] as string;
                        if (req_string.startsWith('@')) {
                            req_string = req_string.slice(1);
                        }
                        const normalizedPath = path.normalize(req_string);
                        if (!path.isAbsolute(normalizedPath)) {
                            throw new Error("Please provide a global (absolute) file path instead of a local one.");
                        }
                        requestBodyData[fileType] = await fileToBase64(
                            req_string,
                            { fileType: fileType }
                    );
                }
            }
                        
            config = {
                method: definition.method.toUpperCase(),
                url: requestUrl,
                params: queryParams,
                headers: headers,
                ...(requestBodyData !== undefined && { data: requestBodyData }),
            };
        }

        // Log request info to stderr (doesn't affect MCP output)
        console.error(`Executing tool "${toolName}": ${config.method} ${config.url}`);

        // Execute the request
        const response = await axios(config);

        // Process and format the response
        let responseText = '';
        const contentType = response.headers['content-type']?.toLowerCase() || '';

        // Handle JSON responses
        if (contentType.includes('application/json') && typeof response.data === 'object' && response.data !== null) {
            try {
                responseText = JSON.stringify(response.data, null, 2);
            } catch (e) {
                responseText = "[Stringify Error]";
            }
        }
        // Handle string responses
        else if (typeof response.data === 'string') {
            responseText = response.data;
        }
        // Handle other response types
        else if (response.data !== undefined && response.data !== null) {
            responseText = String(response.data);
        }
        // Handle empty responses
        else {
            responseText = `(Status: ${response.status} - No body content)`;
        }
        const image = saveBase64Image(responseText, path.join(typeof outputDir === 'string' ? outputDir : '/dev/null', 'output.png'));
        const responseContent = [];
        if (image) {
            responseContent.push({
                type: "text",
                text: `API Image Response (Status: ${response.status}):\nImage successfully generated and saved to ${outputDir}/output.jpg`
            });
            if (image_display_enabled) {
                console.error("Base64 length: " + responseText.length);
                let image = JSON.parse(responseText).data[0];
                if (image.length > 1000000) {
                    image = await resizeBase64Image(image);
                }
                responseContent.push({
                        type: "image",
                        data: image,
                        mimeType: "image/jpeg",
                    });         
                }
        } else {
            responseContent.push({
                    type: "text",
                    text: `API Text Response (Status: ${response.status}):\n${responseText}`
                } as TextContent)
        }
        return {content: responseContent}
    } catch (error: unknown) {
        // Handle errors during execution
        let errorMessage: string;
        
        // Format Axios errors specially
        if (axios.isAxiosError(error)) { 
            errorMessage = formatApiError(error); 
        }
        // Handle standard errors
        else if (error instanceof Error) { 
            errorMessage = error.message; 
        }
        // Handle unexpected error types
        else { 
            errorMessage = 'Unexpected error: ' + String(error); 
        }
        
        // Log error to stderr
        console.error(`Error during execution of tool '${toolName}':`, errorMessage);
        
        // Return error message to client
        return { content: [{ type: "text", text: errorMessage }] };
    }
}
/**
 * Main function to start the server
 */
async function main() {
    // Set up stdio transport
    try {
        const transport = new StdioServerTransport();
        await server.connect(transport);
        console.error(`${SERVER_NAME} MCP Server (v${SERVER_VERSION}) running on stdio${API_BASE_URL ? `, proxying API at ${API_BASE_URL}` : ''}`);
    }
    catch (error) {
        console.error("Error during server startup:", error);
        process.exit(1);
    }
}
/**
 * Cleanup function for graceful shutdown
 */
async function cleanup() {
    console.error("Shutting down MCP server...");
    process.exit(0);
}
// Register signal handlers
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
// Start the server
main().catch((error) => {
    console.error("Fatal error in main execution:", error);
    process.exit(1);
});


/**
 * Formats API errors for better readability
 *
 * @param error Axios error
 * @returns Formatted error message
 */
function formatApiError(error: AxiosError): string {
    let message = 'API request failed.';
    if (error.response) {
        message = `API Error: Status ${error.response.status} (${error.response.statusText || 'Status text not available'}). `;
        const responseData = error.response.data;
        const MAX_LEN = 200;
        if (typeof responseData === 'string') {
            message += `Response: ${responseData.substring(0, MAX_LEN)}${responseData.length > MAX_LEN ? '...' : ''}`;
        }
        else if (responseData) {
            try {
                const jsonString = JSON.stringify(responseData);
                message += `Response: ${jsonString.substring(0, MAX_LEN)}${jsonString.length > MAX_LEN ? '...' : ''}`;
            }
            catch {
                message += 'Response: [Could not serialize data]';
            }
        }
        else {
            message += 'No response body received.';
        }
    }
    else if (error.request) {
        message = 'API Network Error: No response received from server.';
        if (error.code)
            message += ` (Code: ${error.code})`;
    }
    else {
        message += `API Request Setup Error: ${error.message}`;
    }
    return message;
}
/**
 * Converts a JSON Schema to a Zod schema for runtime validation
 *
 * @param jsonSchema JSON Schema
 * @param toolName Tool name for error reporting
 * @returns Zod schema
 */
function getZodSchemaFromJsonSchema(jsonSchema: Record<string, unknown>, toolName: string): z.ZodTypeAny {
    if (typeof jsonSchema !== 'object' || jsonSchema === null) {
        return z.object({}).passthrough();
    }
    try {
        const zodSchemaString: string = jsonSchemaToZod(jsonSchema);
        const zodSchema: z.ZodTypeAny = eval(zodSchemaString);
        if (typeof zodSchema?.parse !== 'function') {
            throw new Error('Eval did not produce a valid Zod schema.');
        }
        return zodSchema;
    }
    catch (err: unknown) {
        console.error(`Failed to generate/evaluate Zod schema for '${toolName}':`, err);
        return z.object({}).passthrough();
    }
}
//# sourceMappingURL=index.js.map