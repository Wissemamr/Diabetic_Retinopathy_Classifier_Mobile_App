package com.example.mydiabclassapp

//Android, Jetpack Compose, and PyTorch functions.
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.layout.ContentScale
import com.example.mydiabclassapp.ui.theme.MyDiabClassAppTheme
import org.pytorch.*
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import kotlin.math.exp


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MyDiabClassAppTheme {
                //main app entry point
                ClassifierScreen()
            }
        }
    }
}

// this composable decorator is used to build the UI with Jetpack Compose
@Composable
fun ClassifierScreen() {
    // defining state vars
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) } //store img URL
    var bitmap by remember { mutableStateOf<Bitmap?>(null) } // the actual image.
    var result by remember { mutableStateOf<Pair<String, Boolean>?>(null) } // classif output (text & color flag).
    var isProcessing by remember { mutableStateOf(false) } //Tracks if classification is running

    // image picker
    val context = LocalContext.current

    val imagePicker = rememberLauncherForActivityResult( //to open the gallery and browse
        contract = ActivityResultContracts.GetContent() //lets the user pick an image
    ) { uri: Uri? ->
        selectedImageUri = uri
        // We reset the classif result so when we browse a new image the old result doesn't stay on the screen
        result = null
        uri?.let {
            val inputStream: InputStream? = context.contentResolver.openInputStream(it)
            bitmap = BitmapFactory.decodeStream(inputStream)
        }
        //when an img is selected, the uri is stored and the bitmap is created from the inp stream
    }
//==================== Defining the User Interface=========================
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Diabetic Retinopathy Classifier",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(10.dp))
                .padding(8.dp),
            contentAlignment = Alignment.Center
        ) {
            bitmap?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "Selected Image",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            } ?: Text(
                text = "Browse to select an image",
                textAlign = TextAlign.Center
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        Button(onClick = { imagePicker.launch("image/*") }) {
            Text("Browse Gallery")
        }

        Spacer(modifier = Modifier.height(10.dp))

        Button(onClick = {
            bitmap?.let {
                isProcessing = true
                result = classifyImage(context, it)
                isProcessing = false
            }
        }) {
            Text("Classify")
        }

        if (isProcessing) {
            CircularProgressIndicator()
        }

        result?.let { (predictionText, isNoDR) ->
            val resultColor = if (isNoDR) Color(34, 139, 34) else Color.Red

            Column(
                modifier = Modifier
                    .padding(top = 16.dp)
                    .background(resultColor.copy(alpha = 0.1f), RoundedCornerShape(10.dp))
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Prediction:",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black // Always black
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = predictionText.split("(")[0].trim(),
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = resultColor, // Class prediction in green or red
                    textAlign = TextAlign.Center
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = predictionText.split("(")[1].replace(")", "").trim(),
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    textAlign = TextAlign.Center
                )
            }
        }
    }
}

fun classifyImage(context: android.content.Context, bitmap: Bitmap): Pair<String, Boolean> {
    // the backbone of the app, loads the classifier model and runs inference.
    return try {
        val modelPath = assetFilePath(context, "mobilenet_cpu_only.pt")
        if (!File(modelPath).exists()) {
            return Pair("Error: Model file not found at $modelPath", false)
        }

        val model = Module.load(modelPath)
        //resizes the image to 224x224 as required by mobilenet v2
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )
        // run inference and gets probabilities.
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray

        val expScores = scores.map { exp(it.toDouble()) }
        // apply softmax for probabilities.
        val sumExpScores = expScores.sum()
        val probabilities = expScores.map { it / sumExpScores }

        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val classes = arrayOf("Diabetic Retinopathy", "No Diabetic Retinopathy")

        val confidence = (probabilities[maxIndex] * 100).toInt()
        val prediction = "${classes[maxIndex]} (Confidence: $confidence%)"

        Pair(prediction, maxIndex == 1)
    } catch (e: Exception) {
        e.printStackTrace()
        Pair("Error: ${e.message}", false)
    }
}


// helper func for copying a file from the app's assets directory to the app's internal storage and returning its absolute path
fun assetFilePath(context: android.content.Context, assetName: String): String {
    val file = File(context.filesDir, assetName)
    if (file.exists() && file.length() > 0) {
        return file.absolutePath
    }
    try {
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                inputStream.copyTo(outputStream)
            }
        }
    } catch (e: Exception) {
        e.printStackTrace()
    }
    return file.absolutePath
}
