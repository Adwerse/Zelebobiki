<script setup>
import { onMounted, onBeforeUnmount, ref, watch } from 'vue'

const props = defineProps({
  deck: { type: Object, required: true }, // передаємо існуючий Deck instance
  coordinates: { type: Array, required: true }, // [lng, lat]
})

const emit = defineEmits(['close'])

const container = ref(null)
const position = ref({ x: 0, y: 0 })

let deck = null

// оновлюємо позицію DOM-елемента відповідно до координат
function updatePosition() {
  if (!deck || !props.coordinates) return
  const [lng, lat] = props.coordinates
  position.value = { x: lng, y: lat }
}

onMounted(() => {
  deck = props.deck

  if (!deck) return

  updatePosition()

  // слухаємо оновлення рендеру
  deck.on('afterRender', updatePosition)
})

onBeforeUnmount(() => {
  if (deck) {
    deck.removeAllListeners('afterRender')
  }
})
</script>

<template>
  <div
    ref="container"
    class="absolute bg-white shadow-lg rounded-xl p-2"
    :style="{
      left: position.x + 'px',
      top: position.y + 'px',
      transform: 'translate(-50%, -100%)'
    }"
  >
    <button
      class="absolute top-1 right-1 text-gray-500 hover:text-black"
      @click="emit('close')"
    >
      ✖
    </button>
    <div class="p-4">
      <slot>Контент попапа</slot>
    </div>
  </div>
</template>

<style scoped>
.absolute {
  position: absolute;
  pointer-events: auto;
}
</style>
