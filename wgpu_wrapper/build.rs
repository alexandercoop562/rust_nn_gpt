fn main() {
    if std::env::var("CI").is_ok() {
        println!("cargo::rustc-cfg=ci");
    }
}
