document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("renderCanvas");
    const engine = new BABYLON.Engine(canvas, true);
    const scene = new BABYLON.Scene(engine);

    // 🌍 Procedural Skybox (No textures required)
    const skyMaterial = new BABYLON.SkyMaterial("skyMaterial", scene);
    skyMaterial.backFaceCulling = false;
    skyMaterial.luminance = 0.9;  // Brightness of the sky
    skyMaterial.turbidity = 5;    // Amount of atmospheric haze
    skyMaterial.rayleigh = 2;     // Scattering effect

    const skybox = BABYLON.MeshBuilder.CreateBox("skyBox", { size: 5000.0 }, scene);
    skybox.material = skyMaterial;

    // 🏞️ Ground with Reflection
    const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: 100, height: 100 }, scene);
    const groundMaterial = new BABYLON.StandardMaterial("groundMat", scene);
    groundMaterial.diffuseColor = new BABYLON.Color3(0.2, 0.2, 0.2);
    groundMaterial.specularColor = new BABYLON.Color3(0.5, 0.5, 0.5); // Reflective effect
    groundMaterial.reflectionTexture = new BABYLON.MirrorTexture("mirror", 1024, scene, true);
    groundMaterial.reflectionTexture.mirrorPlane = new BABYLON.Plane(0, -1, 0, 0);
    ground.material = groundMaterial;

    // 🎥 Camera & Lighting
    const camera = new BABYLON.ArcRotateCamera("camera", Math.PI / 2, Math.PI / 3, 20, BABYLON.Vector3.Zero(), scene);
    camera.attachControl(canvas, true);

    const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(1, 1, 0), scene);
    light.intensity = 0.8;

    // 🚀 Load AI Bot Model
    let bot;
    BABYLON.SceneLoader.ImportMesh("", "static/", "ai_bot.glb", scene, function (meshes) {
        bot = meshes[0];
        bot.position = new BABYLON.Vector3(0, 1, 0);

        // ✅ Idle Hover Animation (Smooth up & down)
        const idleAnimation = new BABYLON.Animation("idleAnimation", "position.y", 30,
            BABYLON.Animation.ANIMATIONTYPE_FLOAT, BABYLON.Animation.ANIMATIONLOOPMODE_CYCLE);

        const keys = [
            { frame: 0, value: 1 },
            { frame: 30, value: 1.3 },
            { frame: 60, value: 1 }
        ];

        idleAnimation.setKeys(keys);
        bot.animations.push(idleAnimation);
        scene.beginAnimation(bot, 0, 60, true); // Loop animation
    });

    // 💥 Explosion Effect (for Auto-Kill)
    function createExplosion(position) {
        let particleSystem = new BABYLON.ParticleSystem("particles", 500, scene);
        particleSystem.particleTexture = new BABYLON.Texture("static/explosion.png", scene);
        particleSystem.emitter = position;
        particleSystem.minSize = 0.2;
        particleSystem.maxSize = 1.5;
        particleSystem.minEmitPower = 2;
        particleSystem.maxEmitPower = 6;
        particleSystem.color1 = new BABYLON.Color4(1, 0.5, 0, 1);
        particleSystem.color2 = new BABYLON.Color4(1, 0, 0, 1);
        particleSystem.gravity = new BABYLON.Vector3(0, -9.81, 0);
        particleSystem.start();

        setTimeout(() => {
            particleSystem.stop();
        }, 500);
    }

    // 🎮 Function to request AI decision from Flask
    function getAIDecision() {
        fetch("/ai_decision")
            .then(response => response.json())
            .then(data => {
                document.getElementById("scenarioText").innerText = "Scenario: " + data.scenario;
                document.getElementById("decisionText").innerText = "AI Decision: " + data.decision;
                document.getElementById("biasText").innerText = "Bias Score: " + data.bias_score.toFixed(2);

                // 🔊 Play decision sound
                document.getElementById("decisionSound").play();

                if (data.biased) {
                    document.getElementById("status").innerHTML = "🚨 <strong>Bias Detected! Bot Auto-Killed!</strong>";
                    document.getElementById("status").style.color = "red";

                    if (bot) {
                        createExplosion(bot.position); // 💥 Explosion Effect
                        document.getElementById("explosionSound").play();

                        setTimeout(() => {
                            bot.dispose();
                        }, 500);
                    }
                } else {
                    document.getElementById("status").innerHTML = "✅ <strong>Decision Passed.</strong>";
                    document.getElementById("status").style.color = "green";

                    if (bot) {
                        // ✅ Decision Action Animation (Jump Effect)
                        const decisionAnimation = new BABYLON.Animation("decisionJump", "position.y", 30,
                            BABYLON.Animation.ANIMATIONTYPE_FLOAT, BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT);

                        const jumpKeys = [
                            { frame: 0, value: bot.position.y },
                            { frame: 10, value: bot.position.y + 0.5 },
                            { frame: 20, value: bot.position.y }
                        ];

                        decisionAnimation.setKeys(jumpKeys);
                        bot.animations.push(decisionAnimation);
                        scene.beginAnimation(bot, 0, 20, false);
                    }
                }
            })
            .catch(error => {
                console.error("❌ Error fetching AI decision:", error);
                document.getElementById("status").innerText = "⚠️ AI Decision Error! Check server.";
            });
    }

    // 🎮 Attach function to button click
    document.getElementById("decisionButton").addEventListener("click", getAIDecision);

    // 🎥 Render the scene
    engine.runRenderLoop(() => scene.render());
    window.addEventListener("resize", () => engine.resize());
});
