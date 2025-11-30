function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('Путь скопирован в буфер обмена!');
    });
}

function openFile(path) {
    if (confirm('Открыть файл: ' + path)) {
        fetch('/open-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({path: path})
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Файл открыт в проводнике');
            } else {
                alert('Ошибка при открытии файла: ' + data.error);
            }
        });
    }
}