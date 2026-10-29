use super::*;

#[test]
fn accept_error_does_not_drop_subsequent_connections() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    client
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();
    let (server, _) = listener.accept().unwrap();
    serve_incoming(
        [
            Err(std::io::Error::from(std::io::ErrorKind::ConnectionAborted)),
            Ok(server),
        ],
        &AtomicBool::new(false),
        |mut socket| {
            socket.write_all(b"ok").unwrap();
        },
    )
    .unwrap();
    let mut response = [0; 2];
    client.read_exact(&mut response).unwrap();
    assert_eq!(&response, b"ok");
}

#[test]
fn stalled_connection_does_not_block_other_clients() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let shutdown = Arc::new(AtomicBool::new(false));
    let stop = shutdown.clone();
    let (started, observed) = std::sync::mpsc::channel();
    let server = std::thread::spawn(move || {
        serve_connections(listener, &stop, |mut stream| {
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            started.send(()).unwrap();
            let mut input = [0];
            if stream.read_exact(&mut input).is_ok() {
                stream.write_all(b"ok").unwrap();
            }
        })
        .unwrap();
    });
    let stalled = TcpStream::connect(address).unwrap();
    observed
        .recv_timeout(std::time::Duration::from_secs(2))
        .unwrap();
    let mut ready = TcpStream::connect(address).unwrap();
    ready
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .unwrap();
    ready.write_all(b"x").unwrap();
    let mut reply = [0; 2];
    ready.read_exact(&mut reply).unwrap();
    assert_eq!(&reply, b"ok");
    shutdown.store(true, Ordering::Relaxed);
    drop(stalled);
    let _ = TcpStream::connect(address); // Wake accept so the scoped workers can join.
    server.join().unwrap();
}
