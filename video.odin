package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import chan "core:sync/chan"
import "core:sync"
import "core:sys/windows"
import "core:thread"

Video_Info :: struct {
	width:    int,
	height:   int,
	fps:      f32,
	n_frames: int,
}

Ffmpeg_Proc :: struct {
	pi:   windows.PROCESS_INFORMATION,
	pipe: windows.HANDLE,
}

Video_In :: struct {
	ffmpeg: Ffmpeg_Proc,
	info:   Video_Info,
	req:    chan.Chan([]u8),
	done:   chan.Chan(bool),
	thread: ^thread.Thread,
}

Video_Out :: struct {
	ffmpeg: Ffmpeg_Proc,
	frames: chan.Chan([]u8),
	thread: ^thread.Thread,
}

spawn_ffmpeg :: proc(cmd: string, use_stdout: bool) -> (Ffmpeg_Proc, bool) {
	read_pipe, write_pipe: windows.HANDLE
	sa: windows.SECURITY_ATTRIBUTES
	sa.nLength = size_of(windows.SECURITY_ATTRIBUTES)
	sa.bInheritHandle = true
	if !windows.CreatePipe(&read_pipe, &write_pipe, &sa, 0) {
		return {}, false
	}
	if use_stdout {
		windows.SetHandleInformation(read_pipe, 0x1, 0)
	} else {
		windows.SetHandleInformation(write_pipe, 0x1, 0)
	}
	cmd_w := utf8_to_wstring_own(cmd)
	defer delete(cmd_w)
	si: windows.STARTUPINFOW
	si.cb = size_of(windows.STARTUPINFOW)
	si.dwFlags = windows.STARTF_USESTDHANDLES
	if use_stdout {
		si.hStdOutput = write_pipe
		si.hStdInput = windows.GetStdHandle(windows.STD_INPUT_HANDLE)
	} else {
		si.hStdInput = read_pipe
		si.hStdOutput = windows.GetStdHandle(windows.STD_OUTPUT_HANDLE)
	}
	si.hStdError = windows.GetStdHandle(windows.STD_ERROR_HANDLE)
	pi: windows.PROCESS_INFORMATION
	if !windows.CreateProcessW(nil, cast(windows.LPCWSTR)raw_data(cmd_w), nil, nil, true, 0, nil, nil, &si, &pi) {
		windows.CloseHandle(read_pipe)
		windows.CloseHandle(write_pipe)
		return {}, false
	}
	if use_stdout {
		windows.CloseHandle(write_pipe)
	} else {
		windows.CloseHandle(read_pipe)
	}
	windows.CloseHandle(pi.hThread)
	if use_stdout {
		return Ffmpeg_Proc{pi = pi, pipe = read_pipe}, true
	}
	return Ffmpeg_Proc{pi = pi, pipe = write_pipe}, true
}

ffmpeg_close :: proc(p: ^Ffmpeg_Proc) {
	if p.pipe != nil {
		windows.CloseHandle(p.pipe)
		p.pipe = nil
	}
	if p.pi.hProcess != nil {
		windows.WaitForSingleObject(p.pi.hProcess, 5000)
		windows.CloseHandle(p.pi.hProcess)
		p.pi.hProcess = nil
	}
}

utf8_to_wstring_own :: proc(s: string) -> []u16 {
	buf := make([]u16, len(s) + 1)
	n := 0
	i := 0
	for i < len(s) {
		c := s[i]
		cp: u32
		if c < 0x80 {
			cp = u32(c)
			i += 1
		} else if c < 0xE0 {
			cp = u32(c & 0x1F) << 6 | u32(s[i + 1] & 0x3F)
			i += 2
		} else if c < 0xF0 {
			cp = u32(c & 0x0F) << 12 | u32(s[i + 1] & 0x3F) << 6 | u32(s[i + 2] & 0x3F)
			i += 3
		} else {
			cp = u32(c & 0x07) << 18 | u32(s[i + 1] & 0x3F) << 12 | u32(s[i + 2] & 0x3F) << 6 | u32(s[i + 3] & 0x3F)
			i += 4
		}
		if cp > 0xFFFF {
			cp -= 0x10000
			buf[n] = u16(0xD800 + (cp >> 10))
			n += 1
			buf[n] = u16(0xDC00 + (cp & 0x3FF))
			n += 1
		} else {
			buf[n] = u16(cp)
			n += 1
		}
	}
	buf[n] = 0
	return buf[:n + 1]
}

run_capture :: proc(cmd: string) -> (string, bool) {
	p, ok := spawn_ffmpeg(cmd, true)
	if !ok {
		return "", false
	}
	defer ffmpeg_close(&p)
	buf: [dynamic]u8
	defer delete(buf)
	for {
		tmp: [4096]u8
		n: u32
		if !windows.ReadFile(p.pipe, &tmp[0], 4096, &n, nil) {
			break
		}
		if n == 0 {
			break
		}
		append(&buf, ..tmp[:n])
	}
	return strings.clone(string(buf[:])), true
}

video_probe :: proc(path: string) -> (Video_Info, bool) {
	cmd := fmt.tprintf(
		"ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1 \"%s\"",
		path,
	)
	out, ok := run_capture(cmd)
	if !ok {
		return {}, false
	}
	defer delete(out)
	info: Video_Info
	lines := strings.split_lines(out)
	defer delete(lines)
	for line in lines {
		kv := strings.split(line, "=")
		if len(kv) != 2 {
			continue
		}
		key := strings.trim_space(kv[0])
		val := strings.trim_space(kv[1])
		switch key {
		case "width":
			if v, ok := strconv.parse_int(val); ok {info.width = v}
		case "height":
			if v, ok := strconv.parse_int(val); ok {info.height = v}
		case "r_frame_rate":
			parts := strings.split(val, "/")
			if len(parts) == 2 {
				num := f32(parse_int_or(parts[0], 0))
				den := f32(max(1, parse_int_or(parts[1], 1)))
				info.fps = num / den
			}
			delete(parts)
		case "nb_frames":
			if v, ok := strconv.parse_int(val); ok {info.n_frames = v}
		}
	}
	if info.width <= 0 || info.height <= 0 {
		return {}, false
	}
	if info.fps <= 0 {
		info.fps = 30
	}
	if info.n_frames <= 0 {
		info.n_frames = -1
	}
	return info, true
}

video_in_start :: proc(path: string, info: Video_Info) -> (^Video_In, bool) {
	v := new(Video_In)
	v.info = info
	cmd := fmt.tprintf(
		"ffmpeg -v error -hide_banner -i \"%s\" -f rawvideo -pix_fmt rgb24 -an -",
		path,
	)
	p, ok := spawn_ffmpeg(cmd, true)
	if !ok {
		free(v)
		return nil, false
	}
	v.ffmpeg = p
	req, cerr := chan.create_buffered(chan.Chan([]u8), 1, context.allocator)
	if cerr != nil {
		ffmpeg_close(&v.ffmpeg)
		free(v)
		return nil, false
	}
	v.req = req
	done, cerr2 := chan.create_buffered(chan.Chan(bool), 1, context.allocator)
	if cerr2 != nil {
		chan.destroy(v.req)
		ffmpeg_close(&v.ffmpeg)
		free(v)
		return nil, false
	}
	v.done = done
	v.thread = thread.create(video_decode_thread)
	v.thread.data = v
	if v.thread == nil {
		chan.destroy(v.done)
		chan.destroy(v.req)
		ffmpeg_close(&v.ffmpeg)
		free(v)
		return nil, false
	}
	thread.start(v.thread)
	return v, true
}

video_decode_thread :: proc(t: ^thread.Thread) {
	v := cast(^Video_In)t.data
	for {
		buf, ok := chan.recv(v.req)
		if !ok {
			break
		}
		if !video_read_frame(&v.ffmpeg, buf) {
			chan.send(v.done, false)
			break
		}
		if !chan.send(v.done, true) {
			break
		}
	}
}

video_read_frame :: proc(p: ^Ffmpeg_Proc, buf: []u8) -> bool {
	done: u32
	for done < u32(len(buf)) {
		n: u32
		if !windows.ReadFile(p.pipe, raw_data(buf[done:]), u32(len(buf)) - done, &n, nil) {
			return false
		}
		if n == 0 {
			return false
		}
		done += n
	}
	return true
}

video_next_frame :: proc(v: ^Video_In, buf: []u8) -> bool {
	if !chan.send(v.req, buf) {
		return false
	}
	ok, _ := chan.recv(v.done)
	return ok
}

video_in_finish :: proc(v: ^Video_In) {
	chan.close(v.req)
	chan.close(v.done)
	thread.destroy(v.thread)
	ffmpeg_close(&v.ffmpeg)
	free(v)
}

video_out_start :: proc(path: string, w: int, h: int, fps: f32, audio_source: string, qp: int) -> (^Video_Out, bool) {
	v := new(Video_Out)
	cmd := fmt.tprintf(
		"ffmpeg -y -v error -hide_banner -i \"%s\" -f rawvideo -pix_fmt rgb24 -s %dx%d -r %.3f -i - -map 1:v:0 -map 0:a? -c:v hevc_nvenc -preset p7 -tune hq -rc constqp -qp %d -c:a aac -b:a 192k -pix_fmt yuv420p \"%s\"",
		audio_source,
		w,
		h,
		fps,
		qp,
		path,
	)
	p, ok := spawn_ffmpeg(cmd, false)
	if !ok {
		free(v)
		return nil, false
	}
	v.ffmpeg = p
	frames, cerr := chan.create_buffered(chan.Chan([]u8), 1, context.allocator)
	if cerr != nil {
		ffmpeg_close(&v.ffmpeg)
		free(v)
		return nil, false
	}
	v.frames = frames
	v.thread = thread.create(video_encode_thread)
	v.thread.data = v
	if v.thread == nil {
		chan.destroy(v.frames)
		ffmpeg_close(&v.ffmpeg)
		free(v)
		return nil, false
	}
	thread.start(v.thread)
	return v, true
}

video_encode_thread :: proc(t: ^thread.Thread) {
	v := cast(^Video_Out)t.data
	for {
		buf, ok := chan.recv(v.frames)
		if !ok {
			break
		}
		video_write_frame(&v.ffmpeg, buf)
		delete(buf)
	}
}

video_write_frame :: proc(p: ^Ffmpeg_Proc, buf: []u8) -> bool {
	done: u32
	for done < u32(len(buf)) {
		n: u32
		if !windows.WriteFile(p.pipe, raw_data(buf[done:]), u32(len(buf)) - done, &n, nil) {
			return false
		}
		if n == 0 {
			return false
		}
		done += n
	}
	return true
}

video_send_encode :: proc(v: ^Video_Out, buf: []u8) -> bool {
	return chan.send(v.frames, buf)
}

video_out_finish :: proc(v: ^Video_Out) {
	chan.close(v.frames)
	thread.destroy(v.thread)
	ffmpeg_close(&v.ffmpeg)
	free(v)
}

parse_int_or :: proc(s: string, def: int) -> int {
	if v, ok := strconv.parse_int(s); ok {
		return v
	}
	return def
}
