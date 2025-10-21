import Foundation
import CoreText
import AppKit

func convertMarkdownToPDF(mdPath: String, pdfPath: String) throws {
    let url = URL(fileURLWithPath: mdPath)
    let markdown = try String(contentsOf: url, encoding: .utf8)
    let attributed: NSAttributedString
    if #available(macOS 12.0, *) {
        let parsed = try AttributedString(markdown: markdown)
        attributed = NSAttributedString(parsed)
    } else {
        attributed = NSAttributedString(string: markdown)
    }
    let mutable = NSMutableAttributedString(attributedString: attributed)
    let fullRange = NSRange(location: 0, length: mutable.length)
    let paragraph = NSMutableParagraphStyle()
    paragraph.lineSpacing = 4
    paragraph.paragraphSpacing = 8
    mutable.addAttribute(.font, value: NSFont.systemFont(ofSize: 12), range: fullRange)
    mutable.addAttribute(.paragraphStyle, value: paragraph, range: fullRange)

    let framesetter = CTFramesetterCreateWithAttributedString(mutable)
    let pdfData = NSMutableData()
    guard let consumer = CGDataConsumer(data: pdfData as CFMutableData) else {
        throw NSError(domain: "MDToPDF", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create PDF consumer"])
    }
    var mediaBox = CGRect(x: 0, y: 0, width: 595, height: 842)
    guard let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil) else {
        throw NSError(domain: "MDToPDF", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create PDF context"])
    }

    var currentLocation: CFIndex = 0
    let pageRect = CGRect(x: 48, y: 48, width: 595 - 96, height: 842 - 96)

    while currentLocation < mutable.length {
        context.beginPDFPage(nil)
        let path = CGMutablePath()
        path.addRect(pageRect)
        let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: currentLocation, length: 0), path, nil)
        CTFrameDraw(frame, context)
        let visibleRange = CTFrameGetVisibleStringRange(frame)
        if visibleRange.length == 0 {
            break
        }
        currentLocation += visibleRange.length
        context.endPDFPage()
    }

    context.closePDF()
    try pdfData.write(to: URL(fileURLWithPath: pdfPath), options: .atomic)
}

let arguments = CommandLine.arguments
if arguments.count != 3 {
    fputs("Usage: md_to_pdf.swift <input.md> <output.pdf>\n", stderr)
    exit(1)
}

do {
    try convertMarkdownToPDF(mdPath: arguments[1], pdfPath: arguments[2])
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
