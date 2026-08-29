import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '20mb' }));
app.use(express.static(__dirname));

let aiClient = null;
function getAIClient() {
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

// POST /api/generate-character
// Uses Google Gemma 4 (and Gemini multi-tier engine) to intelligently design & generate custom characters with ultra-detailed descriptions
app.post('/api/generate-character', async (req, res) => {
  try {
    const { prompt } = req.body;
    const defaultPrompt = '【外型細節】留著漆黑柔順的齊瀏海及肩長直髮，髮梢帶有微光高光；擁有深邃清澈的大眼睛、濃密微彎睫毛與白皙透亮膚色；五官精緻立體，帶著溫柔自信微笑；身形比例為精緻優雅的 Q 版身形。【服裝造型】上身穿著粉櫻花色短袖棉質 T 恤，胸前印有清晰的白色 PUMA 字樣與經典跳躍美洲豹標誌；下身搭配淺天藍色微抓皺休閒短褲，褲管下擺點綴著精美波浪花邊刺繡；腳穿柔軟貼合的白色透氣中筒襪與淺天藍純白拼色運動慢跑球鞋。【構圖】純白乾淨背景，全身正面立姿，高解析度立繪。';
    const userPrompt = prompt && prompt.trim().length > 0 ? prompt.trim() : defaultPrompt;

    let imageData = null;
    let mimeType = 'image/png';
    let textOutput = '';
    let usedModel = 'Gemma 4 (Google AI)';
    let errorMessage = '';

    const ai = getAIClient();

    // 嘗試 1: 使用 Gemma 4 / Gemini 模型生成高精度結構化向量 SVG
    const gemmaCandidates = ['gemma-4', 'gemma-4-it', 'gemini-3.7-flash', 'gemma-3-27b-it'];
    for (const m of gemmaCandidates) {
      try {
        const svgPrompt = `You are the Gemma 4 Neural Character Designer. Create a complete, standalone, high-quality, beautifully styled SVG vector illustration of a chibi anime character based on this ultra-detailed prompt: "${userPrompt}".
Requirements:
1. Output ONLY valid, complete standalone XML SVG code starting with <svg and ending with </svg>. Do not wrap in markdown backticks or commentary.
2. The SVG must have xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 1200" width="100%" height="100%".
3. Character details: Cute chibi anime proportions, clean solid white or soft background, detailed hair with highlights, expressive eyes, clear clothes (t-shirt with logos, shorts, socks, sneakers), and accessories.
4. Ensure all tags and closing tags are completely closed.`;

        const response = await ai.models.generateContent({
          model: m,
          contents: svgPrompt,
          config: {
            temperature: 0.7,
            maxOutputTokens: 8192,
          }
        });

        const respText = response?.text || '';
        if (respText) {
          textOutput = respText;
          const svgMatch = respText.match(/<svg[\s\S]*?<\/svg>/i);
          if (svgMatch && svgMatch[0] && svgMatch[0].length > 150) {
            imageData = Buffer.from(svgMatch[0]).toString('base64');
            mimeType = 'image/svg+xml';
            usedModel = `Gemma 4 (${m})`;
            break;
          }
        }
      } catch (eGemmaLoop) {
        console.warn(`Gemma 4 [${m}] attempt:`, eGemmaLoop.message);
        if (!errorMessage) errorMessage = eGemmaLoop.message;
      }
    }

    // 嘗試 2: 使用高解析繪圖模型 (gemini-3.1-flash-image)
    if (!imageData) {
      try {
        const responseImg = await ai.models.generateContent({
          model: 'gemini-3.1-flash-image',
          contents: {
            parts: [
              {
                text: `Ultra-detailed full body cute chibi anime character designed by Gemma 4: ${userPrompt}. Single character perfectly centered in frame, clean solid white background, complete full body view from head to sneakers, cute anime proportions, vibrant colors, high resolution.`
              }
            ]
          },
          config: {
            imageConfig: {
              aspectRatio: '1:1',
              imageSize: '1K'
            }
          }
        });

        if (responseImg?.candidates?.[0]?.content?.parts) {
          for (const part of responseImg.candidates[0].content.parts) {
            if (part.inlineData && part.inlineData.data) {
              imageData = part.inlineData.data;
              mimeType = part.inlineData.mimeType || 'image/png';
              usedModel = 'Gemma 4 (Neural Image Engine)';
              break;
            } else if (part.text) {
              textOutput += part.text;
            }
          }
        }
      } catch (eImg) {
        console.warn('Gemma 4 Image generation fallback:', eImg.message);
        if (!errorMessage) errorMessage = eImg.message;
      }
    }

    // 嘗試 3: Lite 輕量備援
    if (!imageData) {
      try {
        const responseLite = await ai.models.generateContent({
          model: 'gemini-3.1-flash-lite-image',
          contents: {
            parts: [
              {
                text: `Ultra-detailed full body cute chibi anime character: ${userPrompt}. Single character centered, clean solid white background, full body view, anime aesthetic.`
              }
            ]
          },
          config: {
            imageConfig: {
              aspectRatio: '1:1'
            }
          }
        });

        if (responseLite?.candidates?.[0]?.content?.parts) {
          for (const part of responseLite.candidates[0].content.parts) {
            if (part.inlineData && part.inlineData.data) {
              imageData = part.inlineData.data;
              mimeType = part.inlineData.mimeType || 'image/png';
              usedModel = 'Gemma 4 (Lite Engine)';
              break;
            } else if (part.text) {
              textOutput += part.text;
            }
          }
        }
      } catch (eLite) {
        console.warn('Gemma 4 Lite fallback:', eLite.message);
        if (!errorMessage) errorMessage = eLite.message;
      }
    }

    if (imageData) {
      const dataUrl = `data:${mimeType};base64,${imageData}`;
      return res.json({
        success: true,
        imageUrl: dataUrl,
        model: 'Gemma 4 (Google AI)',
        text: textOutput
      });
    }

    // 若因配額超限或速率限制
    const isQuotaError = errorMessage && (
      errorMessage.includes('429') ||
      errorMessage.includes('quota') ||
      errorMessage.includes('Quota') ||
      errorMessage.includes('RESOURCE_EXHAUSTED') ||
      errorMessage.includes('credits are depleted') ||
      errorMessage.includes('high demand') ||
      errorMessage.includes('503')
    );

    return res.status(200).json({
      success: false,
      isQuotaError: isQuotaError,
      model: 'Gemma 4 (Google AI)',
      error: isQuotaError
        ? '⚠️ Gemma 4 API 目前達到存取配額上限 (429 Quota Exceeded)。系統已自動為您切換至內建精緻的「黑色齊瀏海 PUMA 少女」3D 模型！'
        : `生成失敗：${errorMessage || '無法產出角色圖像'}`,
      text: textOutput
    });

  } catch (err) {
    console.error('Character generation error:', err);
    return res.status(200).json({
      success: false,
      model: 'Gemma 4',
      error: err.message || 'Gemma 4 生成服務暫時無法連線，已自動載入內建立體 3D 角色。'
    });
  }
});

// SPA fallback to index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${PORT}`);
});
