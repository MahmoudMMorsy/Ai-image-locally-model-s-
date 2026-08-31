package com.nanopixel.app

import android.graphics.Bitmap
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var resultImageView: ImageView
    private lateinit var promptEditText: EditText
    private lateinit var statusTextView: TextView
    private lateinit var engine: PixelArtMobileEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultImageView = findViewById(R.id.resultImageView)
        promptEditText = findViewById(R.id.promptEditText)
        statusTextView = findViewById(R.id.statusTextView)

        engine = PixelArtMobileEngine(this)
        engine.loadModel("models/real_latent_unet_256.onnx")

        findViewById<Button>(R.id.generateButton).setOnClickListener {
            val prompt = promptEditText.text.toString()
            if (prompt.isNotEmpty()) {
                statusTextView.text = "جاري التوليد المحلي عبر الـ ONNX..."
                val bitmap = engine.generateSprite(prompt, 256, 256)
                resultImageView.setImageBitmap(bitmap)
                statusTextView.text = "تم التوليد بنجاح!"
            }
        }

        findViewById<Button>(R.id.trainButton).setOnClickListener {
            statusTextView.text = "جاري التدريب المحلي على داتا الموبيل..."
            val loss = engine.localFineTuneOnDevice(filesDir, 5, 0.001f)
            statusTextView.text = "اكتمل التدريب المحلي! Loss النهائي: $loss"
        }
    }
}
