#include <EGL/egl.h>
#include <EGL/eglext.h>
// gl2ext.h needs the core GLES types declared first on NDK 27 and older.
// clang-format off
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
// clang-format on
#include <android/log.h>
#include <android/native_window_jni.h>
#include <android/trace.h>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ffcommon.h"
#include "ffvideo_frame.h"
#include "ffvideo_placebo.h"
#include "ffvideo_surface.h"

extern "C" {
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/pixdesc.h>
}

namespace {

constexpr size_t kMaxQueuedFrames = 2;
constexpr int kRenderThreadNiceValue = -4;
constexpr int kTextureSetCount = 3;
constexpr int kPlaneCount = kVideoPlaneCount;
constexpr size_t kMasteringMetadataValueCount = 10;
constexpr size_t kContentLightMetadataValueCount = 2;
constexpr size_t kMaxDirectBufferCount = 32;
constexpr size_t kRetainedIdleDirectBufferCount = 4;
constexpr size_t kMaxDirectBufferBytes =
    sizeof(void *) >= 8 ? size_t{512} * 1024 * 1024 : size_t{256} * 1024 * 1024;

constexpr EGLint kEglGlColorspace = 0x309D;
constexpr EGLint kEglGlColorspaceBt2020Pq = 0x3340;
constexpr EGLint kEglGlColorspaceBt2020Hlg = 0x3540;
constexpr EGLint kUnknownEglColorspace = -1;
constexpr GLenum kGlR16Ext = 0x822A;

enum class SurfaceColorMode {
  kSdr,
  kBt2020Pq,
  kBt2020Hlg,
};

enum class HdrToSdrTransfer {
  kNone = 0,
  kPq = 1,
  kHlg = 2,
};

const char *SurfaceColorModeName(SurfaceColorMode color_mode) {
  switch (color_mode) {
    case SurfaceColorMode::kBt2020Pq:
      return "BT2020_PQ";
    case SurfaceColorMode::kBt2020Hlg:
      return "BT2020_HLG";
    case SurfaceColorMode::kSdr:
      return "SDR";
  }
  return "unknown";
}

const char *EglColorspaceName(EGLint colorspace) {
  switch (colorspace) {
    case kEglGlColorspaceBt2020Pq:
      return "BT2020_PQ";
    case kEglGlColorspaceBt2020Hlg:
      return "BT2020_HLG";
    case EGL_NONE:
      return "EGL_NONE";
    case kUnknownEglColorspace:
      return "unavailable";
    default:
      return "other";
  }
}

EGLint GetEglColorspace(SurfaceColorMode color_mode) {
  switch (color_mode) {
    case SurfaceColorMode::kBt2020Pq:
      return kEglGlColorspaceBt2020Pq;
    case SurfaceColorMode::kBt2020Hlg:
      return kEglGlColorspaceBt2020Hlg;
    case SurfaceColorMode::kSdr:
      return EGL_NONE;
  }
  return EGL_NONE;
}

class ScopedTrace {
 public:
  explicit ScopedTrace(const char *name) { ATrace_beginSection(name); }
  ~ScopedTrace() { ATrace_endSection(); }
};

bool HasExtension(const char *extensions, const char *extension) {
  if (!extensions || !extension || strchr(extension, ' ')) {
    return false;
  }
  const size_t length = strlen(extension);
  const char *position = extensions;
  while ((position = strstr(position, extension))) {
    const bool starts_at_boundary =
        position == extensions || position[-1] == ' ';
    const bool ends_at_boundary =
        position[length] == '\0' || position[length] == ' ';
    if (starts_at_boundary && ends_at_boundary) {
      return true;
    }
    position += length;
  }
  return false;
}

const char *GlErrorName(GLenum error) {
  switch (error) {
    case GL_INVALID_ENUM:
      return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE:
      return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION:
      return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION:
      return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY:
      return "GL_OUT_OF_MEMORY";
    default:
      return "unknown";
  }
}

bool CheckGlError(const char *operation) {
  bool success = true;
  for (GLenum error; (error = glGetError()) != GL_NO_ERROR;) {
    LOGE("%s failed: %s (0x%x).", operation, GlErrorName(error), error);
    success = false;
  }
  return success;
}

GLuint CompileShader(GLenum type, const char *source) {
  GLuint shader = glCreateShader(type);
  if (!shader) {
    return 0;
  }
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled == GL_TRUE) {
    return shader;
  }
  GLint log_length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
  if (log_length > 1) {
    std::string log(static_cast<size_t>(log_length), '\0');
    glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    LOGE("GLES shader compilation failed: %s", log.c_str());
  }
  glDeleteShader(shader);
  return 0;
}

GLuint CreateProgram(const char *fragment_source, bool use_es31 = false) {
  static constexpr char kVertexShaderEs30[] = R"(#version 300 es
      layout(location = 0) in vec2 in_position;
      layout(location = 1) in vec2 in_tex_coord;
      out vec2 tex_coord;
      uniform int rotation_degrees;
      void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        if (rotation_degrees == 90) {
          tex_coord = vec2(in_tex_coord.y, 1.0 - in_tex_coord.x);
        } else if (rotation_degrees == 180) {
          tex_coord = vec2(1.0 - in_tex_coord.x, 1.0 - in_tex_coord.y);
        } else if (rotation_degrees == 270) {
          tex_coord = vec2(1.0 - in_tex_coord.y, in_tex_coord.x);
        } else {
          tex_coord = in_tex_coord;
        }
      })";
  static constexpr char kVertexShaderEs31[] = R"(#version 310 es
      layout(location = 0) in vec2 in_position;
      layout(location = 1) in vec2 in_tex_coord;
      out vec2 tex_coord;
      uniform int rotation_degrees;
      void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        if (rotation_degrees == 90) {
          tex_coord = vec2(in_tex_coord.y, 1.0 - in_tex_coord.x);
        } else if (rotation_degrees == 180) {
          tex_coord = vec2(1.0 - in_tex_coord.x, 1.0 - in_tex_coord.y);
        } else if (rotation_degrees == 270) {
          tex_coord = vec2(1.0 - in_tex_coord.y, in_tex_coord.x);
        } else {
          tex_coord = in_tex_coord;
        }
      })";

  GLuint vertex_shader = CompileShader(
      GL_VERTEX_SHADER, use_es31 ? kVertexShaderEs31 : kVertexShaderEs30);
  GLuint fragment_shader = CompileShader(GL_FRAGMENT_SHADER, fragment_source);
  if (!vertex_shader || !fragment_shader) {
    if (vertex_shader) {
      glDeleteShader(vertex_shader);
    }
    if (fragment_shader) {
      glDeleteShader(fragment_shader);
    }
    return 0;
  }

  GLuint program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glLinkProgram(program);
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  GLint linked = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) {
    return program;
  }
  GLint log_length = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
  if (log_length > 1) {
    std::string log(static_cast<size_t>(log_length), '\0');
    glGetProgramInfoLog(program, log_length, nullptr, log.data());
    LOGE("GLES program link failed: %s", log.c_str());
  }
  glDeleteProgram(program);
  return 0;
}

SurfaceColorMode GetSurfaceColorMode(const AVFrame *frame) {
  if (frame->color_trc == AVCOL_TRC_SMPTE2084) {
    return SurfaceColorMode::kBt2020Pq;
  }
  if (frame->color_trc == AVCOL_TRC_ARIB_STD_B67) {
    return SurfaceColorMode::kBt2020Hlg;
  }
  return SurfaceColorMode::kSdr;
}

bool CanUseFastSdrShader(const AVFrame *frame) {
  // The fast shader performs YUV matrix/range conversion only. Route frames
  // requiring gamut, transfer or constant-luminance conversion to libplacebo.
  switch (frame->colorspace) {
    case AVCOL_SPC_UNSPECIFIED:
    case AVCOL_SPC_BT709:
    case AVCOL_SPC_FCC:
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
    case AVCOL_SPC_SMPTE240M:
      break;
    default:
      return false;
  }
  switch (frame->color_primaries) {
    case AVCOL_PRI_UNSPECIFIED:
    case AVCOL_PRI_BT709:
    case AVCOL_PRI_BT470BG:
    case AVCOL_PRI_SMPTE170M:
      break;
    default:
      return false;
  }
  switch (frame->color_trc) {
    case AVCOL_TRC_UNSPECIFIED:
    case AVCOL_TRC_BT709:
    case AVCOL_TRC_SMPTE170M:
    case AVCOL_TRC_SMPTE240M:
    case AVCOL_TRC_IEC61966_2_4:
    case AVCOL_TRC_BT1361_ECG:
    case AVCOL_TRC_BT2020_10:
    case AVCOL_TRC_BT2020_12:
      return true;
    default:
      return false;
  }
}

bool CanUseNativeHdrShader(const AVFrame *frame,
                           SurfaceColorMode source_color_mode,
                           SurfaceColorMode output_color_mode,
                           bool apply_dolby_vision_mapping) {
  if (!frame || apply_dolby_vision_mapping ||
      source_color_mode == SurfaceColorMode::kSdr ||
      output_color_mode != source_color_mode ||
      frame->colorspace != AVCOL_SPC_BT2020_NCL ||
      frame->color_primaries != AVCOL_PRI_BT2020 ||
      av_frame_get_side_data(frame, AV_FRAME_DATA_DYNAMIC_HDR_PLUS)) {
    return false;
  }
  return (source_color_mode == SurfaceColorMode::kBt2020Pq &&
          frame->color_trc == AVCOL_TRC_SMPTE2084) ||
         (source_color_mode == SurfaceColorMode::kBt2020Hlg &&
          frame->color_trc == AVCOL_TRC_ARIB_STD_B67);
}

bool ScaleEglHdrMetadata(AVRational value, float minimum, float maximum,
                         EGLint *result) {
  if (!result || value.den == 0) {
    return false;
  }
  const double converted = av_q2d(value);
  if (!std::isfinite(converted) || converted < minimum || converted > maximum) {
    return false;
  }
  const double scaled = converted * EGL_METADATA_SCALING_EXT;
  if (scaled < std::numeric_limits<EGLint>::min() ||
      scaled > std::numeric_limits<EGLint>::max()) {
    return false;
  }
  *result = static_cast<EGLint>(scaled);
  return true;
}

bool GetLumaCoefficients(AVColorSpace colorspace, int width, int height,
                         float *kr, float *kb) {
  switch (colorspace) {
    case AVCOL_SPC_FCC:
      *kr = 0.3000f;
      *kb = 0.1100f;
      break;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
      *kr = 0.2990f;
      *kb = 0.1140f;
      break;
    case AVCOL_SPC_SMPTE240M:
      *kr = 0.2120f;
      *kb = 0.0870f;
      break;
    case AVCOL_SPC_BT709:
      *kr = 0.2126f;
      *kb = 0.0722f;
      break;
    case AVCOL_SPC_BT2020_NCL:
      *kr = 0.2627f;
      *kb = 0.0593f;
      break;
    case AVCOL_SPC_UNSPECIFIED:
      if (width >= 1280 || height >= 720) {
        *kr = 0.2126f;
        *kb = 0.0722f;
      } else {
        *kr = 0.2990f;
        *kb = 0.1140f;
      }
      break;
    default:
      return false;
  }
  return true;
}

struct ColorTransform {
  GLfloat offset[3];
  GLfloat scale[3];
  GLfloat matrix[9];
};

struct ChromaTextureOffset {
  GLfloat x;
  GLfloat y;
};

ChromaTextureOffset GetChromaTextureOffset(const AVFrame *frame,
                                           const YuvFormat &format) {
  ChromaTextureOffset offset = {};
  int x_position;
  int y_position;
  if (av_chroma_location_enum_to_pos(&x_position, &y_position,
                                     frame->chroma_location) < 0) {
    return offset;
  }

  const int chroma_width =
      AV_CEIL_RSHIFT(frame->width, format.chroma_width_shift);
  const int chroma_height =
      AV_CEIL_RSHIFT(frame->height, format.chroma_height_shift);
  if (format.chroma_width_shift > 0) {
    const float subsampling =
        static_cast<float>(1 << format.chroma_width_shift);
    const float centered_position = (subsampling - 1.0f) * 0.5f;
    offset.x = (centered_position - static_cast<float>(x_position) / 256.0f) /
               (subsampling * static_cast<float>(chroma_width));
  }
  if (format.chroma_height_shift > 0) {
    const float subsampling =
        static_cast<float>(1 << format.chroma_height_shift);
    const float centered_position = (subsampling - 1.0f) * 0.5f;
    offset.y = (centered_position - static_cast<float>(y_position) / 256.0f) /
               (subsampling * static_cast<float>(chroma_height));
  }
  return offset;
}

bool GetColorTransform(const AVFrame *frame, int depth,
                       ColorTransform *transform) {
  if (!transform) {
    return false;
  }
  const float max_code = static_cast<float>((UINT32_C(1) << depth) - 1);
  const bool full_range = frame->color_range == AVCOL_RANGE_JPEG ||
                          frame->format == AV_PIX_FMT_YUVJ420P ||
                          frame->format == AV_PIX_FMT_YUVJ422P ||
                          frame->format == AV_PIX_FMT_YUVJ444P;
  const float depth_scale =
      static_cast<float>(UINT32_C(1) << std::max(depth - 8, 0));

  *transform = {};
  transform->offset[0] = full_range ? 0.0f : 16.0f * depth_scale / max_code;
  transform->offset[1] = 128.0f * depth_scale / max_code;
  transform->offset[2] = transform->offset[1];
  transform->scale[0] = full_range ? 1.0f : max_code / (219.0f * depth_scale);
  transform->scale[1] = full_range ? 1.0f : max_code / (224.0f * depth_scale);
  transform->scale[2] = transform->scale[1];

  float kr;
  float kb;
  if (!GetLumaCoefficients(frame->colorspace, frame->width, frame->height, &kr,
                           &kb)) {
    return false;
  }
  const float kg = 1.0f - kr - kb;
  const float red_from_v = 2.0f * (1.0f - kr);
  const float blue_from_u = 2.0f * (1.0f - kb);
  const float green_from_u = -2.0f * kb * (1.0f - kb) / kg;
  const float green_from_v = -2.0f * kr * (1.0f - kr) / kg;

  const GLfloat matrix[9] = {
      1.0f,        1.0f,       1.0f,         0.0f, green_from_u,
      blue_from_u, red_from_v, green_from_v, 0.0f,
  };
  memcpy(transform->matrix, matrix, sizeof(matrix));
  return true;
}

}  // namespace

class VideoSurfaceRenderer::Impl {
 public:
  Impl()
      : direct_frame_pool_(std::make_shared<DirectFramePool>()),
        render_thread_(&Impl::RenderLoop, this) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      ClearQueuedFramesLocked();
      if (requested_window_) {
        ANativeWindow_release(requested_window_);
        requested_window_ = nullptr;
      }
    }
    condition_.notify_all();
    if (render_thread_.joinable()) {
      render_thread_.join();
    }
    if (submitted_window_) {
      ANativeWindow_release(submitted_window_);
      submitted_window_ = nullptr;
    }
  }

  bool Initialize() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(
        lock, [this] { return stop_ || surface_initialization_complete_; });
    return !stop_ && surface_rendering_enabled_.load(std::memory_order_acquire);
  }

  bool IsDirectRenderingEnabled() const {
    return direct_rendering_enabled_.load(std::memory_order_acquire);
  }

  int GetDirectBuffer(AVCodecContext *codec_context, AVFrame *frame, int) {
    DirectFrameLayout layout = {};
    if (!BuildDirectFrameLayout(codec_context, frame, &layout)) {
      if (!unsupported_direct_format_logged_.exchange(
              true, std::memory_order_acq_rel)) {
        const char *pixel_format =
            frame
                ? av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format))
                : nullptr;
        __android_log_print(
            ANDROID_LOG_WARN, LOG_TAG,
            "Direct Surface rendering does not support pixel format %s.",
            pixel_format ? pixel_format : "unknown");
      }
      return AVERROR(ENOSYS);
    }
    if (!direct_rendering_enabled_.load(std::memory_order_acquire) ||
        fatal_error_.load(std::memory_order_acquire)) {
      return AVERROR(ENOSYS);
    }

    std::shared_ptr<DirectFrameSlot> slot =
        direct_frame_pool_->TryAcquire(layout);
    if (!slot) {
      auto request = std::make_shared<DirectAllocationRequest>();
      request->layout = layout;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_ || fatal_error_.load(std::memory_order_acquire)) {
          return AVERROR_EXTERNAL;
        }
        direct_allocation_requests_.push_back(request);
        condition_.notify_all();
        condition_.wait(lock, [this, &request] {
          return request->complete || stop_ ||
                 fatal_error_.load(std::memory_order_acquire);
        });
      }
      slot = request->slot;
      if (!slot) {
        return AVERROR(ENOMEM);
      }
    }

    return BindDirectFrame(direct_frame_pool_, slot, frame);
  }

  bool IsDirectFrame(const AVFrame *frame) const {
    if (!frame || !frame->data[0]) {
      return false;
    }
    const std::shared_ptr<DirectFrameSlot> slot =
        direct_frame_pool_->Find(frame->data[0]);
    if (!slot) {
      return false;
    }
    for (int plane = 1; plane < kPlaneCount; ++plane) {
      if (!frame->data[plane] ||
          direct_frame_pool_->Find(frame->data[plane]) != slot) {
        return false;
      }
    }
    return true;
  }

  bool SupportsFrame(const AVFrame *frame) const {
    YuvFormat format = {};
    return GetYuvFormat(frame, &format);
  }

  VideoRenderResult Render(
      JNIEnv *env, jobject surface, AVFrame *frame, int displayed_width,
      int displayed_height, int64_t release_time_ns, int rotation_degrees,
      DolbyVisionMappingPolicy dolby_vision_mapping_policy) {
    YuvFormat format = {};
    if (!env || !surface || !frame || displayed_width <= 0 ||
        displayed_height <= 0 || displayed_width != frame->width ||
        displayed_height != frame->height ||
        (rotation_degrees != 0 && rotation_degrees != 90 &&
         rotation_degrees != 180 && rotation_degrees != 270) ||
        !GetYuvFormat(frame, &format) ||
        fatal_error_.load(std::memory_order_acquire)) {
      const char *pixel_format =
          frame ? av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format))
                : nullptr;
      LOGE(
          "Unsupported GLES Surface frame: format=%s size=%dx%d display=%dx%d.",
          pixel_format ? pixel_format : "null", frame ? frame->width : 0,
          frame ? frame->height : 0, displayed_width, displayed_height);
      return VideoRenderResult::kError;
    }
    ANativeWindow *window = ANativeWindow_fromSurface(env, surface);
    if (!window) {
      LOGE("Failed to acquire GLES video Surface.");
      return VideoRenderResult::kError;
    }
    const bool replace_surface =
        !submitted_surface_ ||
        !env->IsSameObject(submitted_surface_, surface) ||
        submitted_window_ != window;
    if (replace_surface) {
      jobject surface_ref = env->NewGlobalRef(surface);
      if (!surface_ref) {
        ANativeWindow_release(window);
        LOGE("Failed to retain GLES video Surface.");
        return VideoRenderResult::kError;
      }
      if (submitted_surface_) {
        env->DeleteGlobalRef(submitted_surface_);
      }
      submitted_surface_ = surface_ref;
      if (submitted_window_) {
        ANativeWindow_release(submitted_window_);
      }
      submitted_window_ = window;
      ANativeWindow_acquire(window);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        ClearQueuedFramesLocked();
        if (requested_window_) {
          ANativeWindow_release(requested_window_);
        }
        requested_window_ = window;
        ++requested_generation_;
      }
      condition_.notify_all();
    } else {
      ANativeWindow_release(window);
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_ || fatal_error_.load(std::memory_order_acquire)) {
        return VideoRenderResult::kError;
      }
      if (queued_frames_.size() >= kMaxQueuedFrames) {
        return VideoRenderResult::kTryAgain;
      }
    }

    bool apply_dolby_vision_mapping = false;
    if (dolby_vision_mapping_policy != DolbyVisionMappingPolicy::kDisabled) {
      if (CanMapDolbyVisionMetadata(frame)) {
        apply_dolby_vision_mapping = true;
      } else if (dolby_vision_mapping_policy ==
                 DolbyVisionMappingPolicy::kRequire) {
        LOGE(
            "Required Dolby Vision mapping metadata is unavailable or "
            "invalid.");
        return VideoRenderResult::kError;
      }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_ || fatal_error_.load(std::memory_order_acquire)) {
      return VideoRenderResult::kError;
    }
    if (queued_frames_.size() >= kMaxQueuedFrames) {
      return VideoRenderResult::kTryAgain;
    }
    queued_frames_.emplace_back(frame);
    QueuedFrame &queued_frame = queued_frames_.back();
    queued_frame.window_generation = requested_generation_;
    queued_frame.release_time_ns = release_time_ns;
    queued_frame.rotation_degrees = rotation_degrees;
    queued_frame.apply_dolby_vision_mapping = apply_dolby_vision_mapping;
    lock.unlock();
    condition_.notify_all();
    return VideoRenderResult::kSuccess;
  }

  void Flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    ClearQueuedFramesLocked();
    const uint64_t flush_generation = ++requested_flush_generation_;
    condition_.notify_all();
    condition_.wait(lock, [this, flush_generation] {
      return stop_ || fatal_error_.load(std::memory_order_acquire) ||
             applied_flush_generation_ >= flush_generation;
    });
  }

  void Detach(JNIEnv *env) {
    if (submitted_surface_) {
      env->DeleteGlobalRef(submitted_surface_);
      submitted_surface_ = nullptr;
    }
    if (submitted_window_) {
      ANativeWindow_release(submitted_window_);
      submitted_window_ = nullptr;
    }
    uint64_t detach_generation;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      ClearQueuedFramesLocked();
      if (requested_window_) {
        ANativeWindow_release(requested_window_);
        requested_window_ = nullptr;
      }
      detach_generation = ++requested_generation_;
      condition_.notify_all();
      condition_.wait(lock, [this, detach_generation] {
        return stop_ || fatal_error_.load(std::memory_order_acquire) ||
               completed_generation_ >= detach_generation;
      });
    }
  }

 private:
  struct DirectAllocationRequest {
    DirectFrameLayout layout;
    std::shared_ptr<DirectFrameSlot> slot;
    bool complete = false;
  };

  struct PendingDirectFrame {
    AVFrame *frame = nullptr;
    GLsync fence = nullptr;
  };

  struct QueuedFrame {
    QueuedFrame() = default;
    explicit QueuedFrame(AVFrame *frame) : frame(frame) {}
    QueuedFrame(const QueuedFrame &) = delete;
    QueuedFrame &operator=(const QueuedFrame &) = delete;
    QueuedFrame(QueuedFrame &&other) noexcept { *this = std::move(other); }
    QueuedFrame &operator=(QueuedFrame &&other) noexcept {
      if (this != &other) {
        av_frame_free(&frame);
        frame = std::exchange(other.frame, nullptr);
        window_generation = other.window_generation;
        release_time_ns = other.release_time_ns;
        rotation_degrees = other.rotation_degrees;
        apply_dolby_vision_mapping = other.apply_dolby_vision_mapping;
      }
      return *this;
    }
    ~QueuedFrame() { av_frame_free(&frame); }

    AVFrame *frame = nullptr;
    uint64_t window_generation = 0;
    int64_t release_time_ns = 0;
    int rotation_degrees = 0;
    bool apply_dolby_vision_mapping = false;
  };

  struct TexturePlane {
    GLuint id = 0;
    int width = 0;
    int height = 0;
    GLenum internal_format = 0;
  };

  struct Program {
    GLuint id = 0;
    GLint offset = -1;
    GLint scale = -1;
    GLint matrix = -1;
    GLint max_code = -1;
    GLint normalized_sample_scale = -1;
    GLint chroma_texture_offset = -1;
    GLint rotation_degrees = -1;
    GLint frame_buffer = -1;
    GLint plane_offset = -1;
    GLint plane_stride = -1;
    GLint luma_size = -1;
    GLint chroma_size = -1;
  };

  static constexpr char kFragmentShaderHeader[] = R"(#version 300 es
      precision highp float;
      in vec2 tex_coord;
      layout(location = 0) out vec4 out_color;
      uniform vec3 yuv_offset;
      uniform vec3 yuv_scale;
      uniform mat3 yuv_to_rgb;
      uniform vec2 chroma_texture_offset;
    )";

  static constexpr char kFragmentShaderSample8[] = R"(
      uniform sampler2D y_texture;
      uniform sampler2D u_texture;
      uniform sampler2D v_texture;
      uniform float normalized_sample_scale;
      vec3 sample_yuv() {
        return vec3(texture(y_texture, tex_coord).r,
                    texture(u_texture, tex_coord + chroma_texture_offset).r,
                    texture(v_texture, tex_coord + chroma_texture_offset).r)
               * normalized_sample_scale;
      })";

  static constexpr char kFragmentShaderSamplePacked16[] = R"(
      uniform sampler2D y_texture;
      uniform sampler2D u_texture;
      uniform sampler2D v_texture;
      uniform float max_code;
      float sample_plane(sampler2D plane, vec2 coord) {
        // GL_RG8 stores the little-endian low/high bytes separately. Rebuilding
        // after hardware linear filtering is equivalent to filtering the
        // original 16-bit sample because this operation is linear.
        vec2 bytes = texture(plane, coord).rg;
        return dot(bytes, vec2(255.0, 65280.0)) / max_code;
      }
      vec3 sample_yuv() {
        return vec3(sample_plane(y_texture, tex_coord),
                    sample_plane(u_texture, tex_coord + chroma_texture_offset),
                    sample_plane(v_texture, tex_coord + chroma_texture_offset));
      })";

  static constexpr char kFragmentShaderMain[] = R"(
      void main() {
        vec3 electrical_color =
            yuv_to_rgb * ((sample_yuv() - yuv_offset) * yuv_scale);
        out_color = vec4(electrical_color, 1.0);
      })";

  static constexpr char kDirectBufferShaderHeader[] = R"(#version 310 es
      #extension GL_EXT_texture_buffer : require
      precision highp float;
      precision highp int;
      in vec2 tex_coord;
      layout(location = 0) out vec4 out_color;
      uniform vec3 yuv_offset;
      uniform vec3 yuv_scale;
      uniform mat3 yuv_to_rgb;
      uniform vec2 chroma_texture_offset;
      uniform uvec3 plane_offset;
      uniform ivec3 plane_stride;
      uniform ivec2 luma_size;
      uniform ivec2 chroma_size;

      ivec2 plane_size(int plane) {
        return plane == 0 ? luma_size : chroma_size;
      }

      ivec2 clamp_position(int plane, ivec2 position) {
        return clamp(position, ivec2(0), plane_size(plane) - ivec2(1));
      }
    )";

  static constexpr char kDirectBufferShaderSample8[] = R"(
      uniform highp samplerBuffer frame_buffer;

      float fetch_plane(int plane, ivec2 position) {
        position = clamp_position(plane, position);
        int index = int(plane_offset[plane]) +
                    position.y * plane_stride[plane] + position.x;
        return texelFetch(frame_buffer, index).r;
      }
    )";

  static constexpr char kDirectBufferShaderSample16[] = R"(
      uniform highp usamplerBuffer frame_buffer;
      uniform float max_code;

      float fetch_plane(int plane, ivec2 position) {
        position = clamp_position(plane, position);
        int index = int(plane_offset[plane]) +
                    position.y * plane_stride[plane] + position.x;
        return float(texelFetch(frame_buffer, index).r) / max_code;
      }
    )";

  static constexpr char kDirectBufferShaderMain[] = R"(
      float sample_plane(int plane, vec2 coordinate) {
        vec2 position = coordinate * vec2(plane_size(plane)) - vec2(0.5);
        ivec2 lower = ivec2(floor(position));
        vec2 weight = fract(position);
        float top = mix(fetch_plane(plane, lower),
                        fetch_plane(plane, lower + ivec2(1, 0)), weight.x);
        float bottom = mix(fetch_plane(plane, lower + ivec2(0, 1)),
                           fetch_plane(plane, lower + ivec2(1, 1)), weight.x);
        return mix(top, bottom, weight.y);
      }

      void main() {
        vec2 chroma_coord = tex_coord + chroma_texture_offset;
        vec3 yuv = vec3(sample_plane(0, tex_coord),
                        sample_plane(1, chroma_coord),
                        sample_plane(2, chroma_coord));
        vec3 electrical_color =
            yuv_to_rgb * ((yuv - yuv_offset) * yuv_scale);
        out_color = vec4(electrical_color, 1.0);
      })";

  bool MakeResourceContextCurrent() {
    return egl_display_ != EGL_NO_DISPLAY &&
           resource_egl_context_ != EGL_NO_CONTEXT &&
           resource_egl_surface_ != EGL_NO_SURFACE &&
           eglMakeCurrent(egl_display_, resource_egl_surface_,
                          resource_egl_surface_,
                          resource_egl_context_) == EGL_TRUE;
  }

  bool MakePresentationContextCurrent() {
    return egl_display_ != EGL_NO_DISPLAY && egl_context_ != EGL_NO_CONTEXT &&
           egl_surface_ != EGL_NO_SURFACE &&
           eglMakeCurrent(egl_display_, egl_surface_, egl_surface_,
                          egl_context_) == EGL_TRUE;
  }

  EGLContext CreateEs31Context(EGLConfig config, EGLContext share_context) {
    if (!supports_es31_context_creation_ || !config) {
      return EGL_NO_CONTEXT;
    }
    static constexpr EGLint kContextAttributes[] = {
        EGL_CONTEXT_MAJOR_VERSION_KHR,
        3,
        EGL_CONTEXT_MINOR_VERSION_KHR,
        1,
        EGL_NONE,
    };
    return eglCreateContext(egl_display_, config, share_context,
                            kContextAttributes);
  }

  bool IsCurrentContextEs31() const {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT) {
      return false;
    }
    GLint major_version = 0;
    GLint minor_version = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major_version);
    glGetIntegerv(GL_MINOR_VERSION, &minor_version);
    return major_version > 3 || (major_version == 3 && minor_version >= 1);
  }

  bool IsCurrentContextDirectRenderingCapable() const {
    if (!IsCurrentContextEs31()) {
      return false;
    }
    const char *extensions =
        reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
    return HasExtension(extensions, "GL_EXT_buffer_storage");
  }

  bool ChooseWindowConfig(bool high_precision, EGLConfig *config) {
    if (!config) {
      return false;
    }
    const EGLint component_bits = high_precision ? 10 : 8;
    const EGLint alpha_bits = high_precision ? 2 : 8;
    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE,
        EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT_KHR,
        EGL_RED_SIZE,
        component_bits,
        EGL_GREEN_SIZE,
        component_bits,
        EGL_BLUE_SIZE,
        component_bits,
        EGL_ALPHA_SIZE,
        alpha_bits,
        EGL_NONE,
    };
    EGLint config_count = 0;
    *config = nullptr;
    return eglChooseConfig(egl_display_, config_attributes, config, 1,
                           &config_count) == EGL_TRUE &&
           config_count == 1;
  }

  bool ProbeSharedPresentationContext(EGLConfig config) {
    EGLContext context = CreateEs31Context(config, resource_egl_context_);
    if (context == EGL_NO_CONTEXT) {
      return false;
    }
    eglDestroyContext(egl_display_, context);
    return true;
  }

  bool ProbeDirectBufferStorage() {
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    constexpr GLsizeiptr kProbeSize = 4096;
    constexpr GLbitfield kMapFlags =
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT;
    constexpr GLbitfield kStorageFlags = kMapFlags | GL_CLIENT_STORAGE_BIT_EXT;
    direct_buffer_storage_(GL_PIXEL_UNPACK_BUFFER, kProbeSize, nullptr,
                           kStorageFlags);
    if (!CheckGlError("persistent pixel unpack buffer allocation")) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &buffer);
      return false;
    }
    void *mapping =
        glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, kProbeSize, kMapFlags);
    if (!mapping) {
      CheckGlError("persistent pixel unpack buffer mapping");
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &buffer);
      return false;
    }
    memset(mapping, 0, static_cast<size_t>(kProbeSize));
    glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT);
    const bool unmapped = glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER) == GL_TRUE;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDeleteBuffers(1, &buffer);
    return unmapped && CheckGlError("persistent pixel unpack buffer probe");
  }

  bool InitializeRendererEgl() {
    egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint egl_major_version = 0;
    EGLint egl_minor_version = 0;
    if (egl_display_ == EGL_NO_DISPLAY ||
        eglInitialize(egl_display_, &egl_major_version, &egl_minor_version) !=
            EGL_TRUE ||
        eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) {
      LOGE("Failed to initialize FFmpeg GLES output (0x%x).", eglGetError());
      return false;
    }
    const char *egl_extensions = eglQueryString(egl_display_, EGL_EXTENSIONS);
    supports_smpte2086_metadata_ =
        HasExtension(egl_extensions, "EGL_EXT_surface_SMPTE2086_metadata");
    supports_cta861_3_metadata_ =
        HasExtension(egl_extensions, "EGL_EXT_surface_CTA861_3_metadata");
    supports_es31_context_creation_ =
        egl_major_version > 1 ||
        (egl_major_version == 1 && egl_minor_version >= 5) ||
        HasExtension(egl_extensions, "EGL_KHR_create_context");
    if (!supports_es31_context_creation_) {
      LOGE("EGL does not support explicit OpenGL ES 3.1 contexts.");
      return false;
    }

    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT_KHR,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_NONE,
    };
    EGLint config_count = 0;
    if (eglChooseConfig(egl_display_, config_attributes, &resource_egl_config_,
                        1, &config_count) != EGL_TRUE ||
        config_count != 1) {
      LOGE("Failed to choose FFmpeg GLES resource config (0x%x).",
           eglGetError());
      return false;
    }

    resource_egl_context_ =
        CreateEs31Context(resource_egl_config_, EGL_NO_CONTEXT);
    static constexpr EGLint kPbufferAttributes[] = {
        EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE,
    };
    resource_egl_surface_ = eglCreatePbufferSurface(
        egl_display_, resource_egl_config_, kPbufferAttributes);
    if (resource_egl_context_ == EGL_NO_CONTEXT ||
        resource_egl_surface_ == EGL_NO_SURFACE ||
        !MakeResourceContextCurrent()) {
      LOGE("Failed to create FFmpeg GLES resource context (0x%x).",
           eglGetError());
      return false;
    }

    direct_buffer_storage_ = reinterpret_cast<PFNGLBUFFERSTORAGEEXTPROC>(
        eglGetProcAddress("glBufferStorageEXT"));
    direct_texture_buffer_ = reinterpret_cast<PFNGLTEXBUFFEREXTPROC>(
        eglGetProcAddress("glTexBufferEXT"));
    const char *gl_extensions =
        reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
    const bool direct_rendering_enabled =
        IsCurrentContextDirectRenderingCapable() && direct_buffer_storage_ &&
        ProbeDirectBufferStorage();
    direct_rendering_enabled_.store(direct_rendering_enabled,
                                    std::memory_order_release);
    direct_buffer_sampling_supported_ =
        direct_rendering_enabled && direct_texture_buffer_ &&
        HasExtension(gl_extensions, "GL_EXT_texture_buffer");
    if (!ChooseWindowConfig(/* high_precision= */ false, &sdr_egl_config_) ||
        !ProbeSharedPresentationContext(sdr_egl_config_)) {
      LOGE("EGL cannot share the resource context with an SDR window.");
      return false;
    }
    if (ChooseWindowConfig(/* high_precision= */ true, &hdr_egl_config_) &&
        ProbeSharedPresentationContext(hdr_egl_config_)) {
      hdr_shared_context_supported_ = true;
    } else {
      hdr_egl_config_ = nullptr;
      hdr_shared_context_supported_ = false;
      __android_log_print(
          ANDROID_LOG_WARN, LOG_TAG,
          "EGL cannot share a 10-bit window context; HDR will use the "
          "SDR tone-mapping path.");
    }

    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "FFmpeg GLES Surface output enabled: vendor=%s renderer=%s version=%s.",
        reinterpret_cast<const char *>(glGetString(GL_VENDOR)),
        reinterpret_cast<const char *>(glGetString(GL_RENDERER)),
        reinterpret_cast<const char *>(glGetString(GL_VERSION)));
    if (!direct_rendering_enabled) {
      __android_log_print(
          ANDROID_LOG_WARN, LOG_TAG,
          "Persistent direct rendering is unavailable; using FFmpeg AVFrame "
          "GLES texture uploads.");
    } else if (direct_buffer_sampling_supported_) {
      __android_log_print(
          ANDROID_LOG_INFO, LOG_TAG,
          "FFmpeg zero-upload direct-buffer sampling is available.");
    }
    return true;
  }

  std::shared_ptr<DirectFrameSlot> CreateDirectFrameSlot(
      const DirectFrameLayout &layout) {
    if (!direct_buffer_storage_ || (eglGetCurrentContext() == EGL_NO_CONTEXT &&
                                    !MakeResourceContextCurrent())) {
      return nullptr;
    }
    auto slot = std::make_shared<DirectFrameSlot>();
    slot->layout = layout;
    if (layout.total_size >
            std::numeric_limits<size_t>::max() - (kDirectBufferAlignment - 1) ||
        !AlignSize(layout.total_size + (kDirectBufferAlignment - 1),
                   kDirectBufferAlignment, &slot->buffer_size)) {
      return nullptr;
    }
    if (direct_buffer_count_ >= kMaxDirectBufferCount ||
        direct_buffer_bytes_ > kMaxDirectBufferBytes ||
        slot->buffer_size > kMaxDirectBufferBytes - direct_buffer_bytes_) {
      if (!direct_pool_limit_logged_) {
        LOGE(
            "FFmpeg direct-render pool limit reached "
            "(buffers=%zu/%zu, bytes=%zu/%zu).",
            direct_buffer_count_, kMaxDirectBufferCount, direct_buffer_bytes_,
            kMaxDirectBufferBytes);
        direct_pool_limit_logged_ = true;
      }
      return nullptr;
    }
    glGenBuffers(1, &slot->buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot->buffer);
    constexpr GLbitfield kMapFlags =
        GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT;
    constexpr GLbitfield kStorageFlags = kMapFlags | GL_CLIENT_STORAGE_BIT_EXT;
    direct_buffer_storage_(GL_PIXEL_UNPACK_BUFFER,
                           static_cast<GLsizeiptr>(slot->buffer_size), nullptr,
                           kStorageFlags);
    if (!CheckGlError("direct-render buffer allocation")) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &slot->buffer);
      slot->buffer = 0;
      return nullptr;
    }
    slot->mapped_base =
        glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0,
                         static_cast<GLsizeiptr>(slot->buffer_size), kMapFlags);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    if (!slot->mapped_base) {
      CheckGlError("direct-render buffer mapping");
      glDeleteBuffers(1, &slot->buffer);
      slot->buffer = 0;
      return nullptr;
    }
    const uintptr_t mapping = reinterpret_cast<uintptr_t>(slot->mapped_base);
    const uintptr_t aligned =
        (mapping + kDirectBufferAlignment - 1) & ~(kDirectBufferAlignment - 1);
    const size_t data_offset = static_cast<size_t>(aligned - mapping);
    if (data_offset > slot->buffer_size ||
        layout.total_size > slot->buffer_size - data_offset) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot->buffer);
      glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &slot->buffer);
      slot->buffer = 0;
      slot->mapped_base = nullptr;
      return nullptr;
    }
    slot->data = reinterpret_cast<uint8_t *>(aligned);
    if (!direct_frame_pool_->AddLeasedSlot(slot)) {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot->buffer);
      glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &slot->buffer);
      return nullptr;
    }
    ++direct_buffer_count_;
    direct_buffer_bytes_ += slot->buffer_size;
    return slot;
  }

  void DeleteDirectFrameSlots(
      const std::vector<std::shared_ptr<DirectFrameSlot>> &slots) {
    for (const auto &slot : slots) {
      if (!slot || !slot->buffer) {
        continue;
      }
      const auto texture = direct_buffer_textures_.find(slot->buffer);
      if (texture != direct_buffer_textures_.end()) {
        glDeleteTextures(1, &texture->second);
        direct_buffer_textures_.erase(texture);
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, slot->buffer);
      if (slot->mapped_base) {
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      glDeleteBuffers(1, &slot->buffer);
      if (direct_buffer_count_ > 0) {
        --direct_buffer_count_;
      }
      if (direct_buffer_bytes_ >= slot->buffer_size) {
        direct_buffer_bytes_ -= slot->buffer_size;
      } else {
        direct_buffer_bytes_ = 0;
      }
      slot->buffer = 0;
      slot->mapped_base = nullptr;
      slot->data = nullptr;
    }
  }

  void ProcessDirectAllocationRequest(
      const std::shared_ptr<DirectAllocationRequest> &request) {
    if (eglGetCurrentContext() == EGL_NO_CONTEXT &&
        !MakeResourceContextCurrent()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        request->complete = true;
      }
      condition_.notify_all();
      return;
    }
    active_direct_layout_ = request->layout;
    has_active_direct_layout_ = true;
    DeleteDirectFrameSlots(direct_frame_pool_->TakeIdleSlots(
        request->layout, kRetainedIdleDirectBufferCount));
    std::shared_ptr<DirectFrameSlot> slot =
        direct_frame_pool_->TryAcquire(request->layout);
    if (!slot) {
      slot = CreateDirectFrameSlot(request->layout);
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      request->slot = std::move(slot);
      request->complete = true;
    }
    condition_.notify_all();
  }

  void TrimIdleDirectFrameSlots() {
    if (!has_active_direct_layout_ ||
        (eglGetCurrentContext() == EGL_NO_CONTEXT &&
         !MakeResourceContextCurrent())) {
      return;
    }
    DeleteDirectFrameSlots(direct_frame_pool_->TakeIdleSlots(
        active_direct_layout_, kRetainedIdleDirectBufferCount));
  }

  bool InitializeEglDisplay() {
    if (egl_display_ == EGL_NO_DISPLAY) {
      egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
      if (egl_display_ == EGL_NO_DISPLAY ||
          eglInitialize(egl_display_, nullptr, nullptr) != EGL_TRUE) {
        LOGE("Failed to initialize EGL (0x%x).", eglGetError());
        return false;
      }
    }
    if (display_extensions_initialized_) {
      return true;
    }
    const char *extensions = eglQueryString(egl_display_, EGL_EXTENSIONS);
    supports_bt2020_pq_ =
        HasExtension(extensions, "EGL_EXT_gl_colorspace_bt2020_pq");
    supports_bt2020_hlg_ =
        HasExtension(extensions, "EGL_EXT_gl_colorspace_bt2020_hlg");
    if (HasExtension(extensions, "EGL_ANDROID_presentation_time")) {
      presentation_time_android_ =
          reinterpret_cast<PFNEGLPRESENTATIONTIMEANDROIDPROC>(
              eglGetProcAddress("eglPresentationTimeANDROID"));
    }
    display_extensions_initialized_ = true;
    return true;
  }

  bool EnsureEglContext(bool high_precision) {
    if (!InitializeEglDisplay()) {
      return false;
    }
    if (egl_context_ != EGL_NO_CONTEXT &&
        context_high_precision_ == high_precision) {
      return true;
    }
    if (egl_context_ != EGL_NO_CONTEXT) {
      if (egl_surface_ != EGL_NO_SURFACE &&
          eglGetCurrentContext() == egl_context_) {
        glFinish();
        ReleasePendingDirectFrames();
      }
      DestroyEglSurface();
      if (MakeResourceContextCurrent()) {
        DeleteGlResources();
      } else {
        ResetGlResources();
      }
      eglDestroyContext(egl_display_, egl_context_);
      egl_context_ = EGL_NO_CONTEXT;
    }

    egl_config_ = high_precision ? hdr_egl_config_ : sdr_egl_config_;
    if (!egl_config_ || (high_precision && !hdr_shared_context_supported_)) {
      LOGE("No share-compatible %s GLES 3.1 EGL config.",
           high_precision ? "10-bit" : "8-bit");
      return false;
    }

    egl_context_ = CreateEs31Context(egl_config_, resource_egl_context_);
    if (egl_context_ == EGL_NO_CONTEXT) {
      LOGE("Failed to create shared GLES 3.1 context (0x%x).", eglGetError());
      return false;
    }
    context_high_precision_ = high_precision;
    return true;
  }

  bool IsSurfaceColorModeAvailable(SurfaceColorMode color_mode) const {
    switch (color_mode) {
      case SurfaceColorMode::kBt2020Pq:
        return hdr_shared_context_supported_ && supports_bt2020_pq_ &&
               !bt2020_pq_surface_rejected_;
      case SurfaceColorMode::kBt2020Hlg:
        return hdr_shared_context_supported_ && supports_bt2020_hlg_ &&
               !bt2020_hlg_surface_rejected_;
      case SurfaceColorMode::kSdr:
        return true;
    }
    return false;
  }

  EGLint GetRejectedEglColorspace(SurfaceColorMode color_mode) const {
    switch (color_mode) {
      case SurfaceColorMode::kBt2020Pq:
        return bt2020_pq_rejected_actual_colorspace_;
      case SurfaceColorMode::kBt2020Hlg:
        return bt2020_hlg_rejected_actual_colorspace_;
      case SurfaceColorMode::kSdr:
        return kUnknownEglColorspace;
    }
    return kUnknownEglColorspace;
  }

  void RejectSurfaceColorMode(SurfaceColorMode color_mode,
                              EGLint actual_colorspace) {
    switch (color_mode) {
      case SurfaceColorMode::kBt2020Pq:
        bt2020_pq_surface_rejected_ = true;
        bt2020_pq_rejected_actual_colorspace_ = actual_colorspace;
        break;
      case SurfaceColorMode::kBt2020Hlg:
        bt2020_hlg_surface_rejected_ = true;
        bt2020_hlg_rejected_actual_colorspace_ = actual_colorspace;
        break;
      case SurfaceColorMode::kSdr:
        break;
    }
  }

  void ResetSurfaceColorRejections() {
    bt2020_pq_surface_rejected_ = false;
    bt2020_hlg_surface_rejected_ = false;
    bt2020_pq_rejected_actual_colorspace_ = kUnknownEglColorspace;
    bt2020_hlg_rejected_actual_colorspace_ = kUnknownEglColorspace;
  }

  bool CreateEglSurface(SurfaceColorMode color_mode,
                        EGLint *actual_colorspace) {
    if (actual_colorspace) {
      *actual_colorspace = kUnknownEglColorspace;
    }
    const bool high_precision = color_mode != SurfaceColorMode::kSdr;
    if (!native_window_ || !EnsureEglContext(high_precision)) {
      return false;
    }
    DestroyEglSurface();

    EGLint native_format = 0;
    if (eglGetConfigAttrib(egl_display_, egl_config_, EGL_NATIVE_VISUAL_ID,
                           &native_format) != EGL_TRUE) {
      LOGE("Failed to configure GLES native window.");
      return false;
    }
    const int current_format = ANativeWindow_getFormat(native_window_);
    if (current_format != native_format) {
      // Preserve the SurfaceHolder buffer size selected by ExoPlayer while
      // switching to the EGL config's required pixel format.
      const int buffer_width = ANativeWindow_getWidth(native_window_);
      const int buffer_height = ANativeWindow_getHeight(native_window_);
      if (buffer_width <= 0 || buffer_height <= 0 ||
          ANativeWindow_setBuffersGeometry(native_window_, buffer_width,
                                           buffer_height, native_format) != 0) {
        LOGE("Failed to configure GLES native window geometry.");
        return false;
      }
    }

    const EGLint color_value = GetEglColorspace(color_mode);
    const EGLint default_attributes[] = {EGL_NONE};
    const EGLint color_attributes[] = {
        kEglGlColorspace,
        color_value,
        EGL_NONE,
    };
    egl_surface_ = eglCreateWindowSurface(
        egl_display_, egl_config_, native_window_,
        color_value == EGL_NONE ? default_attributes : color_attributes);
    if (egl_surface_ == EGL_NO_SURFACE) {
      LOGE("Failed to create GLES window Surface (0x%x).", eglGetError());
      DestroyEglSurface();
      return false;
    }
    if (color_value != EGL_NONE) {
      EGLint queried_colorspace = kUnknownEglColorspace;
      if (eglQuerySurface(egl_display_, egl_surface_, kEglGlColorspace,
                          &queried_colorspace) != EGL_TRUE) {
        LOGE("Failed to query GLES window Surface colorspace (0x%x).",
             eglGetError());
        DestroyEglSurface();
        return false;
      }
      if (actual_colorspace) {
        *actual_colorspace = queried_colorspace;
      }
      if (queried_colorspace != color_value) {
        DestroyEglSurface();
        return false;
      }
    } else if (actual_colorspace) {
      *actual_colorspace = EGL_NONE;
    }
    if (eglMakeCurrent(egl_display_, egl_surface_, egl_surface_,
                       egl_context_) != EGL_TRUE) {
      LOGE("Failed to make GLES window Surface current (0x%x).", eglGetError());
      DestroyEglSurface();
      return false;
    }
    if (!IsCurrentContextEs31()) {
      LOGE(
          "Presentation context does not provide the required GLES 3.1 "
          "capabilities.");
      DestroyEglSurface();
      return false;
    }
    eglSwapInterval(egl_display_, 1);
    surface_color_mode_ = color_mode;
    surface_actual_colorspace_ =
        color_value == EGL_NONE ? EGL_NONE : color_value;
    if (EnsureGlResources()) {
      return true;
    }
    DeleteGlResources();
    DestroyEglSurface();
    return false;
  }

  bool EnsureEglSurface(SurfaceColorMode color_mode,
                        EGLint *actual_colorspace = nullptr) {
    if (actual_colorspace) {
      *actual_colorspace = kUnknownEglColorspace;
    }
    if (!native_window_) {
      return false;
    }
    if (egl_surface_ != EGL_NO_SURFACE && surface_color_mode_ == color_mode) {
      if (eglGetCurrentContext() != egl_context_ &&
          !MakePresentationContextCurrent()) {
        LOGE("Failed to restore the GLES presentation context (0x%x).",
             eglGetError());
        return false;
      }
      if (actual_colorspace) {
        *actual_colorspace = surface_actual_colorspace_;
      }
      return true;
    }
    if (!IsSurfaceColorModeAvailable(color_mode)) {
      if (actual_colorspace) {
        *actual_colorspace = GetRejectedEglColorspace(color_mode);
      }
      return false;
    }

    const bool had_previous_surface = egl_surface_ != EGL_NO_SURFACE;
    const SurfaceColorMode previous_color_mode = surface_color_mode_;
    EGLint attempted_actual_colorspace = kUnknownEglColorspace;
    if (CreateEglSurface(color_mode, &attempted_actual_colorspace)) {
      if (actual_colorspace) {
        *actual_colorspace = attempted_actual_colorspace;
      }
      return true;
    }

    if (color_mode != SurfaceColorMode::kSdr) {
      RejectSurfaceColorMode(color_mode, attempted_actual_colorspace);
    }
    if (had_previous_surface && previous_color_mode != color_mode) {
      EGLint restored_actual_colorspace = kUnknownEglColorspace;
      if (!CreateEglSurface(previous_color_mode, &restored_actual_colorspace)) {
        LOGE("Failed to restore previous %s GLES window Surface.",
             SurfaceColorModeName(previous_color_mode));
      }
    }
    if (actual_colorspace) {
      *actual_colorspace = attempted_actual_colorspace;
    }
    return false;
  }

  void LogHdrRouteOnce(SurfaceColorMode requested_color_mode,
                       EGLint actual_colorspace,
                       HdrToSdrTransfer hdr_to_sdr_transfer) {
    const int source_index =
        requested_color_mode == SurfaceColorMode::kBt2020Pq ? 0 : 1;
    const int route_index =
        hdr_to_sdr_transfer == HdrToSdrTransfer::kNone ? 0 : 1;
    if (hdr_route_logged_[source_index][route_index]) {
      return;
    }
    hdr_route_logged_[source_index][route_index] = true;
    const EGLint requested_colorspace = GetEglColorspace(requested_color_mode);
    __android_log_print(
        ANDROID_LOG_INFO, LOG_TAG,
        "GLES HDR route: requested=%s(0x%x) actual=%s(0x%x) final=%s.",
        EglColorspaceName(requested_colorspace), requested_colorspace,
        EglColorspaceName(actual_colorspace), actual_colorspace,
        hdr_to_sdr_transfer == HdrToSdrTransfer::kNone ? "HDR_SURFACE"
                                                       : "SDR_TONE_MAP");
  }

  bool UpdateHdrSurfaceMetadata(const EGLint *attributes, const EGLint *values,
                                size_t value_count, EGLint *cached_values,
                                bool *cache_valid, const char *metadata_name) {
    if (*cache_valid &&
        std::equal(values, values + value_count, cached_values)) {
      return true;
    }
    const bool has_metadata =
        std::any_of(values, values + value_count,
                    [](EGLint value) { return value != EGL_DONT_CARE; });
    if (!*cache_valid && !has_metadata) {
      std::copy(values, values + value_count, cached_values);
      *cache_valid = true;
      return true;
    }
    for (size_t index = 0; index < value_count; ++index) {
      if (eglSurfaceAttrib(egl_display_, egl_surface_, attributes[index],
                           values[index]) != EGL_TRUE) {
        const EGLint error = eglGetError();
        for (size_t restore_index = 0; restore_index < value_count;
             ++restore_index) {
          eglSurfaceAttrib(
              egl_display_, egl_surface_, attributes[restore_index],
              *cache_valid ? cached_values[restore_index] : EGL_DONT_CARE);
        }
        (void)eglGetError();
        LOGE("Failed to apply %s metadata (0x%x).", metadata_name, error);
        return false;
      }
    }
    std::copy(values, values + value_count, cached_values);
    *cache_valid = true;
    return true;
  }

  void ApplyHdrSurfaceMetadata(const AVFrame *frame) {
    if (!frame || egl_surface_ == EGL_NO_SURFACE) {
      return;
    }

    static constexpr EGLint kMasteringAttributes[] = {
        EGL_SMPTE2086_DISPLAY_PRIMARY_RX_EXT,
        EGL_SMPTE2086_DISPLAY_PRIMARY_RY_EXT,
        EGL_SMPTE2086_DISPLAY_PRIMARY_GX_EXT,
        EGL_SMPTE2086_DISPLAY_PRIMARY_GY_EXT,
        EGL_SMPTE2086_DISPLAY_PRIMARY_BX_EXT,
        EGL_SMPTE2086_DISPLAY_PRIMARY_BY_EXT,
        EGL_SMPTE2086_WHITE_POINT_X_EXT,
        EGL_SMPTE2086_WHITE_POINT_Y_EXT,
        EGL_SMPTE2086_MAX_LUMINANCE_EXT,
        EGL_SMPTE2086_MIN_LUMINANCE_EXT,
    };
    static_assert(std::size(kMasteringAttributes) ==
                  kMasteringMetadataValueCount);
    EGLint mastering_values[kMasteringMetadataValueCount];
    std::fill_n(mastering_values, std::size(mastering_values), EGL_DONT_CARE);
    bool has_mastering_metadata = false;
    const AVFrameSideData *mastering_side_data =
        av_frame_get_side_data(frame, AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (mastering_side_data && mastering_side_data->data &&
        mastering_side_data->size >= sizeof(AVMasteringDisplayMetadata)) {
      const auto *mastering =
          reinterpret_cast<const AVMasteringDisplayMetadata *>(
              mastering_side_data->data);
      has_mastering_metadata =
          mastering->has_primaries && mastering->has_luminance &&
          ScaleEglHdrMetadata(mastering->display_primaries[0][0], 0.0f, 1.0f,
                              &mastering_values[0]) &&
          ScaleEglHdrMetadata(mastering->display_primaries[0][1], 0.0f, 1.0f,
                              &mastering_values[1]) &&
          ScaleEglHdrMetadata(mastering->display_primaries[1][0], 0.0f, 1.0f,
                              &mastering_values[2]) &&
          ScaleEglHdrMetadata(mastering->display_primaries[1][1], 0.0f, 1.0f,
                              &mastering_values[3]) &&
          ScaleEglHdrMetadata(mastering->display_primaries[2][0], 0.0f, 1.0f,
                              &mastering_values[4]) &&
          ScaleEglHdrMetadata(mastering->display_primaries[2][1], 0.0f, 1.0f,
                              &mastering_values[5]) &&
          ScaleEglHdrMetadata(mastering->white_point[0], 0.0f, 1.0f,
                              &mastering_values[6]) &&
          ScaleEglHdrMetadata(mastering->white_point[1], 0.0f, 1.0f,
                              &mastering_values[7]) &&
          ScaleEglHdrMetadata(mastering->max_luminance, 0.0f, 10000.0f,
                              &mastering_values[8]) &&
          ScaleEglHdrMetadata(mastering->min_luminance, 0.0f, 10000.0f,
                              &mastering_values[9]);
      if (!has_mastering_metadata) {
        std::fill_n(mastering_values, std::size(mastering_values),
                    EGL_DONT_CARE);
      }
    }
    bool mastering_metadata_applied = false;
    if (supports_smpte2086_metadata_) {
      if (!UpdateHdrSurfaceMetadata(
              kMasteringAttributes, mastering_values,
              std::size(kMasteringAttributes), cached_mastering_metadata_,
              &mastering_metadata_cache_valid_, "SMPTE ST 2086")) {
        supports_smpte2086_metadata_ = false;
      } else {
        mastering_metadata_applied = has_mastering_metadata;
      }
    }

    static constexpr EGLint kContentLightAttributes[] = {
        EGL_CTA861_3_MAX_CONTENT_LIGHT_LEVEL_EXT,
        EGL_CTA861_3_MAX_FRAME_AVERAGE_LEVEL_EXT,
    };
    static_assert(std::size(kContentLightAttributes) ==
                  kContentLightMetadataValueCount);
    EGLint content_light_values[kContentLightMetadataValueCount];
    std::fill_n(content_light_values, std::size(content_light_values),
                EGL_DONT_CARE);
    bool has_content_light_metadata = false;
    const AVFrameSideData *content_light_side_data =
        av_frame_get_side_data(frame, AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    if (content_light_side_data && content_light_side_data->data &&
        content_light_side_data->size >= sizeof(AVContentLightMetadata)) {
      const auto *content_light =
          reinterpret_cast<const AVContentLightMetadata *>(
              content_light_side_data->data);
      has_content_light_metadata =
          content_light->MaxCLL <= 10000 &&
          content_light->MaxFALL <= content_light->MaxCLL;
      if (has_content_light_metadata) {
        content_light_values[0] = static_cast<EGLint>(content_light->MaxCLL *
                                                      EGL_METADATA_SCALING_EXT);
        content_light_values[1] = static_cast<EGLint>(content_light->MaxFALL *
                                                      EGL_METADATA_SCALING_EXT);
      }
    }
    bool content_light_metadata_applied = false;
    if (supports_cta861_3_metadata_) {
      if (!UpdateHdrSurfaceMetadata(
              kContentLightAttributes, content_light_values,
              std::size(kContentLightAttributes),
              cached_content_light_metadata_,
              &content_light_metadata_cache_valid_, "CTA-861.3")) {
        supports_cta861_3_metadata_ = false;
      } else {
        content_light_metadata_applied = has_content_light_metadata;
      }
    }

    if ((mastering_metadata_applied || content_light_metadata_applied) &&
        !hdr_surface_metadata_logged_) {
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "Passing FFmpeg HDR static metadata to EGL.");
      hdr_surface_metadata_logged_ = true;
    }
  }

  bool EnsureGlResources() {
    if (program8_.id && (supports_texture_norm16_ || program_packed16_.id) &&
        vertex_array_ && textures_[0][0].id) {
      return true;
    }
    supports_texture_norm16_ =
        HasExtension(reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS)),
                     "GL_EXT_texture_norm16");
    std::string fragment_shader_8 = std::string(kFragmentShaderHeader) +
                                    kFragmentShaderSample8 +
                                    kFragmentShaderMain;
    program8_.id = CreateProgram(fragment_shader_8.c_str());
    if (!supports_texture_norm16_) {
      std::string fragment_shader_packed16 =
          std::string(kFragmentShaderHeader) + kFragmentShaderSamplePacked16 +
          kFragmentShaderMain;
      program_packed16_.id = CreateProgram(fragment_shader_packed16.c_str());
    }
    if (!program8_.id || (!supports_texture_norm16_ && !program_packed16_.id)) {
      return false;
    }
    InitializeProgram(&program8_);
    if (program_packed16_.id) {
      InitializeProgram(&program_packed16_);
    }
    if (direct_buffer_sampling_supported_) {
      const std::string fragment_shader_buffer8 =
          std::string(kDirectBufferShaderHeader) + kDirectBufferShaderSample8 +
          kDirectBufferShaderMain;
      const std::string fragment_shader_buffer16 =
          std::string(kDirectBufferShaderHeader) + kDirectBufferShaderSample16 +
          kDirectBufferShaderMain;
      program_buffer8_.id =
          CreateProgram(fragment_shader_buffer8.c_str(), /* use_es31= */ true);
      program_buffer16_.id =
          CreateProgram(fragment_shader_buffer16.c_str(), /* use_es31= */ true);
      if (program_buffer8_.id && program_buffer16_.id) {
        InitializeDirectBufferProgram(&program_buffer8_);
        InitializeDirectBufferProgram(&program_buffer16_);
      } else {
        if (program_buffer8_.id) {
          glDeleteProgram(program_buffer8_.id);
        }
        if (program_buffer16_.id) {
          glDeleteProgram(program_buffer16_.id);
        }
        program_buffer8_ = {};
        program_buffer16_ = {};
        direct_buffer_sampling_supported_ = false;
        __android_log_print(
            ANDROID_LOG_WARN, LOG_TAG,
            "Disabling FFmpeg zero-upload path because its GLES shaders "
            "could not be created.");
      }
    }

    static constexpr GLfloat kVertices[] = {
        -1.0f, 1.0f, 0.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f,
        1.0f,  1.0f, 1.0f, 0.0f, 1.0f,  -1.0f, 1.0f, 1.0f,
    };
    glGenVertexArrays(1, &vertex_array_);
    glBindVertexArray(vertex_array_);
    glGenBuffers(1, &vertex_buffer_);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                          nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat),
                          reinterpret_cast<void *>(2 * sizeof(GLfloat)));

    for (auto &texture_set : textures_) {
      GLuint texture_ids[kPlaneCount] = {};
      glGenTextures(kPlaneCount, texture_ids);
      for (int plane = 0; plane < kPlaneCount; ++plane) {
        texture_set[plane].id = texture_ids[plane];
      }
    }
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    return CheckGlError("GLES resource creation");
  }

  void InitializeProgram(Program *program) {
    glUseProgram(program->id);
    glUniform1i(glGetUniformLocation(program->id, "y_texture"), 0);
    glUniform1i(glGetUniformLocation(program->id, "u_texture"), 1);
    glUniform1i(glGetUniformLocation(program->id, "v_texture"), 2);
    program->offset = glGetUniformLocation(program->id, "yuv_offset");
    program->scale = glGetUniformLocation(program->id, "yuv_scale");
    program->matrix = glGetUniformLocation(program->id, "yuv_to_rgb");
    program->max_code = glGetUniformLocation(program->id, "max_code");
    program->normalized_sample_scale =
        glGetUniformLocation(program->id, "normalized_sample_scale");
    program->chroma_texture_offset =
        glGetUniformLocation(program->id, "chroma_texture_offset");
    program->rotation_degrees =
        glGetUniformLocation(program->id, "rotation_degrees");
  }

  void InitializeDirectBufferProgram(Program *program) {
    glUseProgram(program->id);
    program->offset = glGetUniformLocation(program->id, "yuv_offset");
    program->scale = glGetUniformLocation(program->id, "yuv_scale");
    program->matrix = glGetUniformLocation(program->id, "yuv_to_rgb");
    program->max_code = glGetUniformLocation(program->id, "max_code");
    program->chroma_texture_offset =
        glGetUniformLocation(program->id, "chroma_texture_offset");
    program->rotation_degrees =
        glGetUniformLocation(program->id, "rotation_degrees");
    program->frame_buffer = glGetUniformLocation(program->id, "frame_buffer");
    program->plane_offset = glGetUniformLocation(program->id, "plane_offset");
    program->plane_stride = glGetUniformLocation(program->id, "plane_stride");
    program->luma_size = glGetUniformLocation(program->id, "luma_size");
    program->chroma_size = glGetUniformLocation(program->id, "chroma_size");
  }

  GLuint GetDirectBufferTexture(const std::shared_ptr<DirectFrameSlot> &slot,
                                const YuvFormat &format) {
    if (!direct_buffer_sampling_supported_ || !direct_texture_buffer_ ||
        !slot || !slot->buffer || format.bytes_per_sample <= 0) {
      return 0;
    }
    const auto existing = direct_buffer_textures_.find(slot->buffer);
    if (existing != direct_buffer_textures_.end()) {
      return existing->second;
    }

    GLint maximum_texels = 0;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE_EXT, &maximum_texels);
    if (maximum_texels <= 0 ||
        slot->buffer_size / static_cast<size_t>(format.bytes_per_sample) >
            static_cast<size_t>(maximum_texels)) {
      return 0;
    }

    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, texture);
    direct_texture_buffer_(GL_TEXTURE_BUFFER_EXT,
                           format.bytes_per_sample == 1 ? GL_R8 : GL_R16UI,
                           slot->buffer);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
    if (!texture || !CheckGlError("direct-buffer texture view")) {
      if (texture) {
        glDeleteTextures(1, &texture);
      }
      return 0;
    }
    direct_buffer_textures_.emplace(slot->buffer, texture);
    return texture;
  }

  bool RenderDirectBuffer(const AVFrame *frame, const YuvFormat &format,
                          const ColorTransform &color_transform,
                          const std::shared_ptr<DirectFrameSlot> &slot,
                          GLuint texture, int target_width, int target_height,
                          int rotation_degrees) {
    if (!texture || !slot->mapped_base) {
      return false;
    }

    GLuint plane_offsets[kPlaneCount] = {};
    GLint plane_strides[kPlaneCount] = {};
    const uintptr_t mapping = reinterpret_cast<uintptr_t>(slot->mapped_base);
    for (int plane = 0; plane < kPlaneCount; ++plane) {
      const uintptr_t data = reinterpret_cast<uintptr_t>(frame->data[plane]);
      if (data < mapping ||
          (data - mapping) % static_cast<uintptr_t>(format.bytes_per_sample) !=
              0) {
        return false;
      }
      const size_t offset =
          static_cast<size_t>(data - mapping) / format.bytes_per_sample;
      const size_t stride =
          static_cast<size_t>(frame->linesize[plane]) / format.bytes_per_sample;
      if (offset > std::numeric_limits<GLuint>::max() ||
          stride > static_cast<size_t>(std::numeric_limits<GLint>::max())) {
        return false;
      }
      plane_offsets[plane] = static_cast<GLuint>(offset);
      plane_strides[plane] = static_cast<GLint>(stride);
    }

    Program &program =
        format.bytes_per_sample == 1 ? program_buffer8_ : program_buffer16_;
    if (!program.id) {
      return false;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, target_width, target_height);
    glUseProgram(program.id);
    glUniform3fv(program.offset, 1, color_transform.offset);
    glUniform3fv(program.scale, 1, color_transform.scale);
    glUniformMatrix3fv(program.matrix, 1, GL_FALSE, color_transform.matrix);
    glUniform1i(program.rotation_degrees, rotation_degrees);
    glUniform3uiv(program.plane_offset, 1, plane_offsets);
    glUniform3iv(program.plane_stride, 1, plane_strides);
    glUniform2i(program.luma_size, frame->width, frame->height);
    glUniform2i(program.chroma_size,
                AV_CEIL_RSHIFT(frame->width, format.chroma_width_shift),
                AV_CEIL_RSHIFT(frame->height, format.chroma_height_shift));
    if (program.max_code >= 0) {
      glUniform1f(program.max_code,
                  static_cast<float>((UINT32_C(1) << format.depth) - 1));
    }
    const ChromaTextureOffset chroma_texture_offset =
        GetChromaTextureOffset(frame, format);
    glUniform2f(program.chroma_texture_offset, chroma_texture_offset.x,
                chroma_texture_offset.y);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, texture);
    glUniform1i(program.frame_buffer, 0);
    glBindVertexArray(vertex_array_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
    return CheckGlError("zero-upload direct-buffer rendering");
  }

  bool UploadPlane(int texture_set, int plane, const AVFrame *frame,
                   const YuvFormat &format,
                   const std::shared_ptr<DirectFrameSlot> &direct_frame_slot,
                   bool use_placebo) {
    const int width =
        plane == 0 ? frame->width
                   : AV_CEIL_RSHIFT(frame->width, format.chroma_width_shift);
    const int height =
        plane == 0 ? frame->height
                   : AV_CEIL_RSHIFT(frame->height, format.chroma_height_shift);
    TexturePlane &texture = textures_[texture_set][plane];
    glActiveTexture(GL_TEXTURE0 + plane);
    glBindTexture(GL_TEXTURE_2D, texture.id);

    // R16UI is core in GLES 3 and lets libplacebo validate the capability it
    // actually needs (sampling) when wrapping the texture.
    const bool normalized_16 = format.bytes_per_sample == 2 &&
                               supports_texture_norm16_ && !use_placebo;
    const bool packed_16 = format.bytes_per_sample == 2 &&
                           !supports_texture_norm16_ && !use_placebo;
    const bool integer_16 = format.bytes_per_sample == 2 && use_placebo;
    const GLenum internal_format = format.bytes_per_sample == 1 ? GL_R8
                                   : normalized_16              ? kGlR16Ext
                                   : integer_16                 ? GL_R16UI
                                                                : GL_RG8;
    const GLenum data_format = packed_16    ? GL_RG
                               : integer_16 ? GL_RED_INTEGER
                                            : GL_RED;
    const GLenum data_type =
        normalized_16 || integer_16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
    const bool allocate_texture = texture.width != width ||
                                  texture.height != height ||
                                  texture.internal_format != internal_format;
    if (allocate_texture) {
      const GLenum filter = integer_16 ? GL_NEAREST : GL_LINEAR;
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                      static_cast<GLint>(filter));
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                      static_cast<GLint>(filter));
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(internal_format), width,
                   height, 0, data_format, data_type, nullptr);
      texture.width = width;
      texture.height = height;
      texture.internal_format = internal_format;
      if (!CheckGlError("GLES texture allocation")) {
        return false;
      }
    }

    if (static_cast<size_t>(width) >
        std::numeric_limits<size_t>::max() /
            static_cast<size_t>(format.bytes_per_sample)) {
      return false;
    }
    const size_t row_bytes =
        static_cast<size_t>(width) * format.bytes_per_sample;
    const size_t row_stride = frame->linesize[plane];
    const void *source_data = frame->data[plane];
    if (direct_frame_slot) {
      if (!direct_frame_slot->mapped_base) {
        return false;
      }
      const uintptr_t mapping =
          reinterpret_cast<uintptr_t>(direct_frame_slot->mapped_base);
      const uintptr_t source = reinterpret_cast<uintptr_t>(frame->data[plane]);
      if (source < mapping ||
          source - mapping >= direct_frame_slot->buffer_size) {
        return false;
      }
      const size_t source_offset = source - mapping;
      const size_t preceding_rows = static_cast<size_t>(height - 1);
      if (preceding_rows >
              (std::numeric_limits<size_t>::max() - row_bytes) / row_stride ||
          preceding_rows * row_stride + row_bytes >
              direct_frame_slot->buffer_size - source_offset) {
        return false;
      }
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, direct_frame_slot->buffer);
      source_data = reinterpret_cast<const void *>(source_offset);
    } else {
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    glPixelStorei(GL_UNPACK_ROW_LENGTH,
                  frame->linesize[plane] / format.bytes_per_sample);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, data_format,
                    data_type, source_data);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    int &validated_uploads = direct_frame_slot ? validated_direct_uploads_
                                               : validated_avframe_uploads_;
    if (validated_uploads < kPlaneCount) {
      if (!CheckGlError(direct_frame_slot ? "direct-render texture upload"
                                          : "AVFrame texture upload")) {
        return false;
      }
      ++validated_uploads;
      if (validated_uploads == kPlaneCount) {
        __android_log_print(
            ANDROID_LOG_INFO, LOG_TAG,
            direct_frame_slot
                ? "Using FFmpeg zero-CPU-copy direct rendering with one GPU "
                  "texture upload."
                : "Using FFmpeg AVFrame GLES texture uploads.");
      }
    }
    return true;
  }

  bool RenderFrame(const AVFrame *frame, int64_t release_time_ns,
                   int rotation_degrees, bool apply_dolby_vision_mapping,
                   GLsync *direct_upload_fence) {
    if (direct_upload_fence) {
      *direct_upload_fence = nullptr;
    }
    YuvFormat format = {};
    if (!GetYuvFormat(frame, &format) || !InitializeEglDisplay()) {
      return false;
    }
    if (apply_dolby_vision_mapping && !dolby_vision_mapping_logged_) {
      __android_log_print(
          ANDROID_LOG_INFO, LOG_TAG,
          "Using FFmpeg RPU metadata with libplacebo OpenGL Dolby Vision "
          "mapping.");
      dolby_vision_mapping_logged_ = true;
    }
    const SurfaceColorMode source_color_mode = apply_dolby_vision_mapping
                                                   ? SurfaceColorMode::kBt2020Pq
                                                   : GetSurfaceColorMode(frame);
    bool use_placebo = source_color_mode != SurfaceColorMode::kSdr ||
                       !CanUseFastSdrShader(frame);
    if (source_color_mode == SurfaceColorMode::kSdr && use_placebo &&
        !sdr_color_management_logged_) {
      __android_log_print(
          ANDROID_LOG_INFO, LOG_TAG,
          "Using libplacebo OpenGL for SDR color-space conversion.");
      sdr_color_management_logged_ = true;
    }
    SurfaceColorMode output_color_mode = SurfaceColorMode::kSdr;
    HdrToSdrTransfer hdr_to_sdr_transfer = HdrToSdrTransfer::kNone;
    EGLint actual_hdr_colorspace = GetRejectedEglColorspace(source_color_mode);
    if ((source_color_mode == SurfaceColorMode::kBt2020Pq ||
         source_color_mode == SurfaceColorMode::kBt2020Hlg) &&
        IsSurfaceColorModeAvailable(source_color_mode)) {
      output_color_mode = source_color_mode;
    } else if (source_color_mode == SurfaceColorMode::kBt2020Pq) {
      hdr_to_sdr_transfer = HdrToSdrTransfer::kPq;
    } else if (source_color_mode == SurfaceColorMode::kBt2020Hlg) {
      hdr_to_sdr_transfer = HdrToSdrTransfer::kHlg;
    }
    EGLint *actual_colorspace = output_color_mode == SurfaceColorMode::kSdr
                                    ? nullptr
                                    : &actual_hdr_colorspace;
    if (!EnsureEglSurface(output_color_mode, actual_colorspace)) {
      if (output_color_mode == SurfaceColorMode::kSdr ||
          source_color_mode == SurfaceColorMode::kSdr) {
        return false;
      }
      if (source_color_mode == SurfaceColorMode::kBt2020Pq) {
        hdr_to_sdr_transfer = HdrToSdrTransfer::kPq;
      } else {
        hdr_to_sdr_transfer = HdrToSdrTransfer::kHlg;
      }
      output_color_mode = SurfaceColorMode::kSdr;
      if (!EnsureEglSurface(output_color_mode)) {
        return false;
      }
    }
    if (source_color_mode != SurfaceColorMode::kSdr) {
      LogHdrRouteOnce(source_color_mode, actual_hdr_colorspace,
                      hdr_to_sdr_transfer);
    }

    const bool use_native_hdr_shader =
        CanUseNativeHdrShader(frame, source_color_mode, output_color_mode,
                              apply_dolby_vision_mapping);
    if (use_native_hdr_shader) {
      use_placebo = false;
    }
    ColorTransform color_transform = {};
    if (!use_placebo &&
        !GetColorTransform(frame, format.depth, &color_transform)) {
      const char *colorspace = av_color_space_name(frame->colorspace);
      LOGE("Unsupported YUV colorspace for GLES rendering: %s.",
           colorspace ? colorspace : "unknown");
      return false;
    }

    const std::shared_ptr<DirectFrameSlot> direct_frame_slot =
        IsDirectFrame(frame) ? direct_frame_pool_->Find(frame->data[0])
                             : nullptr;
    const GLuint direct_buffer_texture =
        !use_placebo && direct_frame_slot
            ? GetDirectBufferTexture(direct_frame_slot, format)
            : 0;
    const bool use_direct_buffer = direct_buffer_texture != 0;
    if (direct_frame_slot) {
      glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT);
    }
    if (!use_direct_buffer) {
      texture_set_index_ = (texture_set_index_ + 1) % kTextureSetCount;
      {
        ScopedTrace trace("ffmpegGlesUpload");
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        for (int plane = 0; plane < kPlaneCount; ++plane) {
          if (!UploadPlane(texture_set_index_, plane, frame, format,
                           direct_frame_slot, use_placebo)) {
            return false;
          }
        }
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
      }
    }

    // Flutter SurfaceProducer 的 ANativeWindow 默认 buffer 尺寸为 1x1（新 Texture
    // 机制不再自动匹配 widget 尺寸，producer 负责协商——MediaCodec 硬解即自行
    // set_buffers_dimensions）。按帧尺寸（旋转换算后）协商 buffer，否则
    // eglQuerySurface 恒 1x1：每帧只画一个像素放大全屏成纯色闪屏（实锤）。
    const bool rotated = rotation_degrees == 90 || rotation_degrees == 270;
    const int buffer_width = rotated ? frame->height : frame->width;
    const int buffer_height = rotated ? frame->width : frame->height;
    if (buffer_width > 0 && buffer_height > 0 &&
        (ANativeWindow_getWidth(native_window_) != buffer_width ||
         ANativeWindow_getHeight(native_window_) != buffer_height) &&
        ANativeWindow_setBuffersGeometry(
            native_window_, buffer_width, buffer_height,
            ANativeWindow_getFormat(native_window_)) != 0) {
      LOGE("Failed to resize GLES native window to %dx%d.", buffer_width,
           buffer_height);
      return false;
    }

    EGLint width = 0;
    EGLint height = 0;
    if (eglQuerySurface(egl_display_, egl_surface_, EGL_WIDTH, &width) !=
            EGL_TRUE ||
        eglQuerySurface(egl_display_, egl_surface_, EGL_HEIGHT, &height) !=
            EGL_TRUE ||
        width <= 0 || height <= 0) {
      LOGE("Failed to query GLES Surface dimensions.");
      return false;
    }
    if (width != logged_surface_width_ || height != logged_surface_height_) {
      logged_surface_width_ = width;
      logged_surface_height_ = height;
      __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                          "FFmpeg GLES Surface buffer: %dx%d color=%s.", width,
                          height, SurfaceColorModeName(output_color_mode));
    }

    if (use_placebo) {
      PlaceboTexturePlane planes[kPlaneCount];
      for (int plane = 0; plane < kPlaneCount; ++plane) {
        const TexturePlane &texture = textures_[texture_set_index_][plane];
        planes[plane] = {
            texture.id,
            texture.width,
            texture.height,
            static_cast<int>(texture.internal_format),
        };
      }
      PlaceboOutputColorMode placebo_output_mode;
      switch (output_color_mode) {
        case SurfaceColorMode::kBt2020Pq:
          placebo_output_mode = PlaceboOutputColorMode::kBt2020Pq;
          break;
        case SurfaceColorMode::kBt2020Hlg:
          placebo_output_mode = PlaceboOutputColorMode::kBt2020Hlg;
          break;
        case SurfaceColorMode::kSdr:
          placebo_output_mode = PlaceboOutputColorMode::kSdr;
          break;
      }
      ScopedTrace trace("ffmpegPlaceboRender");
      if (!placebo_renderer_.Initialize(egl_display_, egl_context_) ||
          !placebo_renderer_.Render(frame, planes, texture_set_index_, width,
                                    height, rotation_degrees,
                                    placebo_output_mode,
                                    apply_dolby_vision_mapping)) {
        return false;
      }
    } else if (use_direct_buffer) {
      ScopedTrace trace("ffmpegGlesZeroUpload");
      if (!RenderDirectBuffer(frame, format, color_transform, direct_frame_slot,
                              direct_buffer_texture, width, height,
                              rotation_degrees)) {
        return false;
      }
      if (!zero_upload_logged_) {
        __android_log_print(
            ANDROID_LOG_INFO, LOG_TAG,
            "Using FFmpeg zero-upload GLES direct-buffer rendering.");
        zero_upload_logged_ = true;
      }
      if (use_native_hdr_shader && !native_hdr_zero_upload_logged_) {
        __android_log_print(
            ANDROID_LOG_INFO, LOG_TAG,
            "Using FFmpeg native HDR zero-upload rendering for %s.",
            SurfaceColorModeName(source_color_mode));
        native_hdr_zero_upload_logged_ = true;
      }
    } else {
      const bool packed_texture =
          format.bytes_per_sample == 2 && !supports_texture_norm16_;
      Program &program = packed_texture ? program_packed16_ : program8_;
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDisable(GL_BLEND);
      glDisable(GL_CULL_FACE);
      glDisable(GL_DEPTH_TEST);
      glDisable(GL_SCISSOR_TEST);
      glDisable(GL_STENCIL_TEST);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glViewport(0, 0, width, height);
      glUseProgram(program.id);
      glUniform3fv(program.offset, 1, color_transform.offset);
      glUniform3fv(program.scale, 1, color_transform.scale);
      glUniformMatrix3fv(program.matrix, 1, GL_FALSE, color_transform.matrix);
      glUniform1i(program.rotation_degrees, rotation_degrees);
      if (program.max_code >= 0) {
        glUniform1f(program.max_code,
                    static_cast<float>((UINT32_C(1) << format.depth) - 1));
      }
      if (program.normalized_sample_scale >= 0) {
        const float normalized_sample_scale =
            format.bytes_per_sample == 1
                ? 1.0f
                : 65535.0f /
                      static_cast<float>((UINT32_C(1) << format.depth) - 1);
        glUniform1f(program.normalized_sample_scale, normalized_sample_scale);
      }
      const ChromaTextureOffset chroma_texture_offset =
          GetChromaTextureOffset(frame, format);
      glUniform2f(program.chroma_texture_offset, chroma_texture_offset.x,
                  chroma_texture_offset.y);
      glBindVertexArray(vertex_array_);
      {
        ScopedTrace trace("ffmpegGlesRender");
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      }
    }
    if (output_color_mode != SurfaceColorMode::kSdr) {
      ApplyHdrSurfaceMetadata(frame);
    }
    {
      ScopedTrace trace("ffmpegGlesPresent");
      if (presentation_time_android_ &&
          presentation_time_android_(egl_display_, egl_surface_,
                                     release_time_ns) != EGL_TRUE) {
        LOGE("Failed to set GLES frame presentation time (0x%x).",
             eglGetError());
        presentation_time_android_ = nullptr;
      }
      if (eglSwapBuffers(egl_display_, egl_surface_) != EGL_TRUE) {
        LOGE("Failed to present GLES frame (0x%x).", eglGetError());
        return false;
      }
    }
    if (direct_frame_slot && direct_upload_fence) {
      *direct_upload_fence =
          glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, /* flags= */ 0);
      if (!*direct_upload_fence) {
        CheckGlError("direct-render upload fence");
        return false;
      }
      glFlush();
    }
    return true;
  }

  void PollPendingDirectFrames() {
    if (pending_direct_frames_.empty() ||
        eglGetCurrentContext() == EGL_NO_CONTEXT) {
      return;
    }
    for (auto iterator = pending_direct_frames_.begin();
         iterator != pending_direct_frames_.end();) {
      const GLenum wait_result =
          glClientWaitSync(iterator->fence, /* flags= */ 0, /* timeout= */ 0);
      if (wait_result == GL_ALREADY_SIGNALED ||
          wait_result == GL_CONDITION_SATISFIED) {
        glDeleteSync(iterator->fence);
        av_frame_free(&iterator->frame);
        iterator = pending_direct_frames_.erase(iterator);
      } else if (wait_result == GL_WAIT_FAILED) {
        CheckGlError("direct-render upload fence wait");
        glFinish();
        ReleasePendingDirectFrames();
        return;
      } else {
        ++iterator;
      }
    }
  }

  void ReleasePendingDirectFrames() {
    const bool has_context = eglGetCurrentContext() != EGL_NO_CONTEXT;
    for (PendingDirectFrame &pending_frame : pending_direct_frames_) {
      if (pending_frame.fence && has_context) {
        glDeleteSync(pending_frame.fence);
      }
      av_frame_free(&pending_frame.frame);
    }
    pending_direct_frames_.clear();
  }

  void RenderLoop() {
    pthread_setname_np(pthread_self(), "ffmpeg-gles");
    if (setpriority(PRIO_PROCESS, gettid(), kRenderThreadNiceValue) != 0) {
      __android_log_print(ANDROID_LOG_WARN, LOG_TAG,
                          "Failed to raise GLES render thread priority: %s.",
                          strerror(errno));
    }
    const bool surface_rendering_enabled = InitializeRendererEgl();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      surface_rendering_enabled_.store(surface_rendering_enabled,
                                       std::memory_order_release);
      surface_initialization_complete_ = true;
    }
    condition_.notify_all();

    for (;;) {
      QueuedFrame queued_frame = {};
      std::shared_ptr<DirectAllocationRequest> allocation_request;
      ANativeWindow *requested_window = nullptr;
      bool window_changed = false;
      uint64_t flush_generation = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto has_work = [this] {
          return stop_ || !direct_allocation_requests_.empty() ||
                 applied_generation_ != requested_generation_ ||
                 applied_flush_generation_ != requested_flush_generation_ ||
                 (!fatal_error_.load(std::memory_order_acquire) &&
                  !queued_frames_.empty());
        };
        if (pending_direct_frames_.empty()) {
          condition_.wait(lock, has_work);
        } else {
          condition_.wait_for(lock, std::chrono::milliseconds(2), has_work);
        }
        if (stop_) {
          for (const auto &request : direct_allocation_requests_) {
            request->complete = true;
          }
          direct_allocation_requests_.clear();
          lock.unlock();
          condition_.notify_all();
          DestroyEgl();
          return;
        }
        if (!direct_allocation_requests_.empty()) {
          allocation_request = direct_allocation_requests_.front();
          direct_allocation_requests_.pop_front();
        } else {
          if (applied_generation_ != requested_generation_) {
            requested_window = requested_window_;
            requested_window_ = nullptr;
            applied_generation_ = requested_generation_;
            window_changed = true;
          }
          if (applied_flush_generation_ != requested_flush_generation_) {
            flush_generation = requested_flush_generation_;
          } else if (!queued_frames_.empty()) {
            queued_frame = std::move(queued_frames_.front());
            queued_frames_.pop_front();
          }
        }
      }
      condition_.notify_all();

      PollPendingDirectFrames();
      TrimIdleDirectFrameSlots();
      if (allocation_request) {
        if (fatal_error_.load(std::memory_order_acquire)) {
          {
            std::lock_guard<std::mutex> lock(mutex_);
            allocation_request->complete = true;
          }
          condition_.notify_all();
        } else {
          ProcessDirectAllocationRequest(allocation_request);
        }
        continue;
      }
      if (window_changed) {
        ReplaceWindow(requested_window);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          completed_generation_ = applied_generation_;
        }
        condition_.notify_all();
      }
      if (flush_generation) {
        if (egl_surface_ != EGL_NO_SURFACE &&
            MakePresentationContextCurrent()) {
          placebo_renderer_.Flush();
          glFinish();
        } else if (egl_display_ != EGL_NO_DISPLAY &&
                   eglGetCurrentContext() != EGL_NO_CONTEXT) {
          glFinish();
        }
        ReleasePendingDirectFrames();
        TrimIdleDirectFrameSlots();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          applied_flush_generation_ = flush_generation;
        }
        condition_.notify_all();
        continue;
      }
      if (!queued_frame.frame) {
        continue;
      }
      if (queued_frame.window_generation != applied_generation_) {
        av_frame_free(&queued_frame.frame);
        continue;
      }
      const bool has_native_window = native_window_ != nullptr;
      GLsync direct_upload_fence = nullptr;
      const bool rendered =
          !has_native_window ||
          RenderFrame(queued_frame.frame, queued_frame.release_time_ns,
                      queued_frame.rotation_degrees,
                      queued_frame.apply_dolby_vision_mapping,
                      &direct_upload_fence);
      if (rendered && direct_upload_fence) {
        pending_direct_frames_.push_back(
            {queued_frame.frame, direct_upload_fence});
        queued_frame.frame = nullptr;
      }
      if (!rendered && eglGetCurrentContext() != EGL_NO_CONTEXT) {
        glFinish();
      }
      if (direct_upload_fence) {
        if (!rendered) {
          glDeleteSync(direct_upload_fence);
        }
      }
      av_frame_free(&queued_frame.frame);
      if (!rendered) {
        LOGE("GLES video rendering failed.");
        ReleasePendingDirectFrames();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          fatal_error_.store(true, std::memory_order_release);
          surface_rendering_enabled_.store(false, std::memory_order_release);
          direct_rendering_enabled_.store(false, std::memory_order_release);
          ClearQueuedFramesLocked();
        }
        condition_.notify_all();
        continue;
      }
      PollPendingDirectFrames();
      TrimIdleDirectFrameSlots();
    }
  }

  void ReplaceWindow(ANativeWindow *window) {
    DestroyEglSurface();
    if (native_window_) {
      ANativeWindow_release(native_window_);
    }
    native_window_ = window;
    ResetSurfaceColorRejections();
  }

  void DestroyEglSurface() {
    logged_surface_width_ = 0;
    logged_surface_height_ = 0;
    mastering_metadata_cache_valid_ = false;
    content_light_metadata_cache_valid_ = false;
    if (egl_display_ == EGL_NO_DISPLAY) {
      return;
    }
    if (egl_surface_ != EGL_NO_SURFACE && egl_context_ != EGL_NO_CONTEXT) {
      if (eglGetCurrentContext() != egl_context_ &&
          !MakePresentationContextCurrent()) {
        LOGE(
            "Failed to bind the GLES presentation context for teardown "
            "(0x%x).",
            eglGetError());
        placebo_renderer_.Abandon();
      }
      if (eglGetCurrentContext() == egl_context_) {
        glFinish();
        ReleasePendingDirectFrames();
        placebo_renderer_.Shutdown();
      }
    }
    if (resource_egl_context_ != EGL_NO_CONTEXT &&
        resource_egl_surface_ != EGL_NO_SURFACE) {
      if (!MakeResourceContextCurrent()) {
        LOGE("Failed to restore GLES resource context (0x%x).", eglGetError());
      }
    } else {
      eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                     EGL_NO_CONTEXT);
    }
    if (egl_surface_ != EGL_NO_SURFACE) {
      eglDestroySurface(egl_display_, egl_surface_);
      egl_surface_ = EGL_NO_SURFACE;
    }
    surface_color_mode_ = SurfaceColorMode::kSdr;
    surface_actual_colorspace_ = kUnknownEglColorspace;
  }

  void DestroyEgl() {
    if (egl_surface_ != EGL_NO_SURFACE &&
        eglGetCurrentContext() == egl_context_) {
      glFinish();
      ReleasePendingDirectFrames();
    }
    DestroyEglSurface();
    if (resource_egl_context_ != EGL_NO_CONTEXT &&
        MakeResourceContextCurrent()) {
      ReleasePendingDirectFrames();
      DeleteGlResources();
    } else {
      ResetGlResources();
    }
    if (egl_display_ != EGL_NO_DISPLAY && egl_context_ != EGL_NO_CONTEXT) {
      eglDestroyContext(egl_display_, egl_context_);
      egl_context_ = EGL_NO_CONTEXT;
    }
    if (resource_egl_context_ != EGL_NO_CONTEXT &&
        MakeResourceContextCurrent()) {
      glFinish();
      ReleasePendingDirectFrames();
      DeleteDirectFrameSlots(direct_frame_pool_->StopAndTakeSlots());
    } else {
      direct_frame_pool_->StopAndTakeSlots();
      ReleasePendingDirectFrames();
    }
    if (egl_display_ != EGL_NO_DISPLAY) {
      eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE,
                     EGL_NO_CONTEXT);
      if (resource_egl_surface_ != EGL_NO_SURFACE) {
        eglDestroySurface(egl_display_, resource_egl_surface_);
        resource_egl_surface_ = EGL_NO_SURFACE;
      }
      if (resource_egl_context_ != EGL_NO_CONTEXT) {
        eglDestroyContext(egl_display_, resource_egl_context_);
        resource_egl_context_ = EGL_NO_CONTEXT;
      }
    }
    if (egl_display_ != EGL_NO_DISPLAY) {
      eglTerminate(egl_display_);
      egl_display_ = EGL_NO_DISPLAY;
    }
    if (native_window_) {
      ANativeWindow_release(native_window_);
      native_window_ = nullptr;
    }
  }

  void DeleteGlResources() {
    const GLuint programs[] = {
        program8_.id,
        program_packed16_.id,
        program_buffer8_.id,
        program_buffer16_.id,
    };
    for (GLuint program : programs) {
      if (program) {
        glDeleteProgram(program);
      }
    }
    if (vertex_array_ && eglGetCurrentContext() == egl_context_) {
      glDeleteVertexArrays(1, &vertex_array_);
    }
    if (vertex_buffer_) {
      glDeleteBuffers(1, &vertex_buffer_);
    }
    for (auto &texture_set : textures_) {
      for (TexturePlane &texture : texture_set) {
        if (texture.id) {
          glDeleteTextures(1, &texture.id);
        }
      }
    }
    ResetGlResources();
  }

  void ResetGlResources() {
    program8_ = {};
    program_packed16_ = {};
    program_buffer8_ = {};
    program_buffer16_ = {};
    vertex_array_ = 0;
    vertex_buffer_ = 0;
    for (auto &texture_set : textures_) {
      for (TexturePlane &texture : texture_set) {
        texture = {};
      }
    }
    texture_set_index_ = -1;
    supports_texture_norm16_ = false;
    validated_direct_uploads_ = 0;
    validated_avframe_uploads_ = 0;
  }

  void ClearQueuedFramesLocked() { queued_frames_.clear(); }

  EGLDisplay egl_display_ = EGL_NO_DISPLAY;
  EGLConfig egl_config_ = nullptr;
  EGLContext egl_context_ = EGL_NO_CONTEXT;
  EGLSurface egl_surface_ = EGL_NO_SURFACE;
  EGLConfig resource_egl_config_ = nullptr;
  EGLContext resource_egl_context_ = EGL_NO_CONTEXT;
  EGLSurface resource_egl_surface_ = EGL_NO_SURFACE;
  EGLConfig sdr_egl_config_ = nullptr;
  EGLConfig hdr_egl_config_ = nullptr;
  SurfaceColorMode surface_color_mode_ = SurfaceColorMode::kSdr;
  EGLint surface_actual_colorspace_ = kUnknownEglColorspace;
  EGLint bt2020_pq_rejected_actual_colorspace_ = kUnknownEglColorspace;
  EGLint bt2020_hlg_rejected_actual_colorspace_ = kUnknownEglColorspace;
  bool supports_es31_context_creation_ = false;
  bool hdr_shared_context_supported_ = false;
  bool display_extensions_initialized_ = false;
  bool context_high_precision_ = false;
  bool supports_bt2020_pq_ = false;
  bool supports_bt2020_hlg_ = false;
  bool supports_smpte2086_metadata_ = false;
  bool supports_cta861_3_metadata_ = false;
  bool bt2020_pq_surface_rejected_ = false;
  bool bt2020_hlg_surface_rejected_ = false;
  bool supports_texture_norm16_ = false;
  bool hdr_route_logged_[2][2] = {};
  bool dolby_vision_mapping_logged_ = false;
  bool sdr_color_management_logged_ = false;
  bool zero_upload_logged_ = false;
  bool native_hdr_zero_upload_logged_ = false;
  bool hdr_surface_metadata_logged_ = false;
  EGLint cached_mastering_metadata_[kMasteringMetadataValueCount] = {};
  EGLint cached_content_light_metadata_[kContentLightMetadataValueCount] = {};
  bool mastering_metadata_cache_valid_ = false;
  bool content_light_metadata_cache_valid_ = false;
  EGLint logged_surface_width_ = 0;
  EGLint logged_surface_height_ = 0;
  int validated_direct_uploads_ = 0;
  int validated_avframe_uploads_ = 0;
  PFNEGLPRESENTATIONTIMEANDROIDPROC presentation_time_android_ = nullptr;
  PFNGLBUFFERSTORAGEEXTPROC direct_buffer_storage_ = nullptr;
  PFNGLTEXBUFFEREXTPROC direct_texture_buffer_ = nullptr;
  bool direct_buffer_sampling_supported_ = false;

  Program program8_;
  Program program_packed16_;
  Program program_buffer8_;
  Program program_buffer16_;
  GLuint vertex_array_ = 0;
  GLuint vertex_buffer_ = 0;
  TexturePlane textures_[kTextureSetCount][kPlaneCount];
  int texture_set_index_ = -1;
  std::unordered_map<GLuint, GLuint> direct_buffer_textures_;
  PlaceboVideoRenderer placebo_renderer_;

  ANativeWindow *native_window_ = nullptr;
  ANativeWindow *submitted_window_ = nullptr;
  jobject submitted_surface_ = nullptr;
  std::atomic<bool> fatal_error_{false};
  std::atomic<bool> surface_rendering_enabled_{false};
  std::atomic<bool> direct_rendering_enabled_{false};
  std::atomic<bool> unsupported_direct_format_logged_{false};

  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<QueuedFrame> queued_frames_;
  std::deque<std::shared_ptr<DirectAllocationRequest>>
      direct_allocation_requests_;
  std::vector<PendingDirectFrame> pending_direct_frames_;
  ANativeWindow *requested_window_ = nullptr;
  uint64_t requested_generation_ = 0;
  uint64_t applied_generation_ = 0;
  uint64_t completed_generation_ = 0;
  uint64_t requested_flush_generation_ = 0;
  uint64_t applied_flush_generation_ = 0;
  size_t direct_buffer_count_ = 0;
  size_t direct_buffer_bytes_ = 0;
  DirectFrameLayout active_direct_layout_ = {};
  bool has_active_direct_layout_ = false;
  bool direct_pool_limit_logged_ = false;
  bool surface_initialization_complete_ = false;
  bool stop_ = false;
  std::shared_ptr<DirectFramePool> direct_frame_pool_;
  std::thread render_thread_;
};

VideoSurfaceRenderer::VideoSurfaceRenderer()
    : impl_(std::make_unique<Impl>()) {}

VideoSurfaceRenderer::~VideoSurfaceRenderer() = default;

bool VideoSurfaceRenderer::Initialize() { return impl_->Initialize(); }

bool VideoSurfaceRenderer::IsDirectRenderingEnabled() const {
  return impl_->IsDirectRenderingEnabled();
}

int VideoSurfaceRenderer::GetDirectBuffer(AVCodecContext *codec_context,
                                          AVFrame *frame, int flags) {
  return impl_->GetDirectBuffer(codec_context, frame, flags);
}

bool VideoSurfaceRenderer::IsDirectFrame(const AVFrame *frame) const {
  return impl_->IsDirectFrame(frame);
}

bool VideoSurfaceRenderer::SupportsFrame(const AVFrame *frame) const {
  return impl_->SupportsFrame(frame);
}

VideoRenderResult VideoSurfaceRenderer::Render(
    JNIEnv *env, jobject surface, AVFrame *frame, int displayed_width,
    int displayed_height, int64_t release_time_ns, int rotation_degrees,
    DolbyVisionMappingPolicy dolby_vision_mapping_policy) {
  return impl_->Render(env, surface, frame, displayed_width, displayed_height,
                       release_time_ns, rotation_degrees,
                       dolby_vision_mapping_policy);
}

void VideoSurfaceRenderer::Flush() { impl_->Flush(); }

void VideoSurfaceRenderer::Detach(JNIEnv *env) { impl_->Detach(env); }
