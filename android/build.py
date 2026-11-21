"""Build the StepCue Android app without Gradle.

    .venv/Scripts/python android/build.py

Needs a JDK (javac, keytool) and the Android SDK pieces: platforms/android-36/android.jar and
build-tools (aapt2, d8, zipalign, apksigner). The SDK folder comes from ANDROID_HOME or
%LOCALAPPDATA%/StepCueAndroidSdk. Output: companion/stepcue.apk (served by companion_server
at /stepcue.apk). The signing keystore android/stepcue.jks is made on first build and is
gitignored; keep it to ship updates that install over the old app.
"""
import glob
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "build")
APK = os.path.join(ROOT, "companion", "stepcue.apk")
KEYSTORE = os.path.join(HERE, "stepcue.jks")
STOREPASS = "stepcue-local"        # local debug-style key for a sideloaded prototype, not a secret


def sdk():
    for d in (os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT"),
              os.path.join(os.environ.get("LOCALAPPDATA", ""), "StepCueAndroidSdk")):
        if d and os.path.isdir(d):
            return d
    sys.exit("Android SDK not found (set ANDROID_HOME)")


def tool(name, sdkdir):
    hits = sorted(glob.glob(os.path.join(sdkdir, "build-tools", "*", name + "*")))
    hits = [h for h in hits if os.path.splitext(h)[1].lower() in (".exe", ".bat", "")]
    if not hits:
        sys.exit(f"missing build tool {name}")
    return hits[-1]


def jdk(name):
    p = shutil.which(name)
    if p:
        return p
    for d in glob.glob(r"C:\Program Files\Eclipse Adoptium\jdk-*\bin") + glob.glob(r"C:\Program Files\Java\*\bin"):
        c = os.path.join(d, name + ".exe")
        if os.path.isfile(c):
            return c
    sys.exit(f"{name} not found (install a JDK)")


def run(cmd):
    print("  $", os.path.basename(cmd[0]), *[os.path.basename(c) if os.path.isabs(c) else c for c in cmd[1:6]],
          "..." if len(cmd) > 6 else "")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        sys.exit(f"failed: {os.path.basename(cmd[0])}")
    return r


def main():
    s = sdk()
    jar = sorted(glob.glob(os.path.join(s, "platforms", "android-*", "android.jar")))[-1]
    aapt2, d8, zipalign, apksigner = (tool(n, s) for n in ("aapt2", "d8", "zipalign", "apksigner"))
    javac, keytool = jdk("javac"), jdk("keytool")
    shutil.rmtree(OUT, ignore_errors=True)
    for d in ("res", "gen", "classes", "dex"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    print("1/6 resources")
    run([aapt2, "compile", "--dir", os.path.join(HERE, "res"), "-o", os.path.join(OUT, "res", "res.zip")])
    base = os.path.join(OUT, "base.apk")
    run([aapt2, "link", "-I", jar, "--manifest", os.path.join(HERE, "AndroidManifest.xml"),
         "--java", os.path.join(OUT, "gen"), "-o", base, os.path.join(OUT, "res", "res.zip")])

    print("2/6 java")
    srcs = glob.glob(os.path.join(HERE, "src", "**", "*.java"), recursive=True)
    srcs += glob.glob(os.path.join(OUT, "gen", "**", "*.java"), recursive=True)
    run([javac, "-nowarn", "-Xlint:-options", "-encoding", "UTF-8", "-source", "11", "-target", "11",
         "-classpath", jar, "-d", os.path.join(OUT, "classes"), *srcs])

    print("3/6 dex")
    classes = glob.glob(os.path.join(OUT, "classes", "**", "*.class"), recursive=True)
    run([d8, "--release", "--min-api", "26", "--lib", jar, "--output", os.path.join(OUT, "dex"), *classes])

    print("4/6 package")
    unsigned = os.path.join(OUT, "unsigned.apk")
    shutil.copy(base, unsigned)
    with zipfile.ZipFile(unsigned, "a", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(OUT, "dex", "classes.dex"), "classes.dex")
    aligned = os.path.join(OUT, "aligned.apk")
    run([zipalign, "-f", "-p", "4", unsigned, aligned])

    print("5/6 sign")
    if not os.path.isfile(KEYSTORE):
        run([keytool, "-genkeypair", "-keystore", KEYSTORE, "-storepass", STOREPASS, "-keypass", STOREPASS,
             "-alias", "stepcue", "-keyalg", "RSA", "-keysize", "2048", "-validity", "10000",
             "-dname", "CN=StepCue prototype"])
    run([apksigner, "sign", "--ks", KEYSTORE, "--ks-pass", "pass:" + STOREPASS, "--ks-key-alias", "stepcue",
         "--out", APK, aligned])

    print("6/6 verify")
    run([apksigner, "verify", APK])
    print(f"built {os.path.relpath(APK, ROOT)} ({os.path.getsize(APK) // 1024} KB)")


if __name__ == "__main__":
    main()
