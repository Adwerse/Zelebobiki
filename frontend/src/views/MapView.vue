<script setup>
import { Deck, TileLayer, BitmapLayer, MapView, HeatmapLayer } from 'deck.gl';
import { onBeforeUnmount, onMounted, onUnmounted, ref, watch } from 'vue'

const deckContainer = ref(null);
const coords = ref(null);
const deck = ref(null);
const popUp = ref(null);

const INITIAL_VIEW_STATE = {
  longitude: 0,
  latitude: 0,
  zoom: 2,
  pitch: 0,
  bearing: 0
};

onMounted(() => {
  document.body.classList.add('no-scroll')
})
onUnmounted(() => {
  document.body.classList.remove('no-scroll')
})

const oceanBasemap = new TileLayer({
  data: 'https://services.arcgisonline.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
  minZoom: 0,
  maxZoom: 10,
  tileSize: 256,

  renderSubLayers: props => {
    const {
      bbox: {west, south, east, north}
    } = props.tile;
    return new BitmapLayer(props, {
      data: null,
      image: props.data,
      bounds: [west, south, east, north]
    });
  }
});

const layers = ref([oceanBasemap])

const loadHeatmap = async () => {
  // const response = await axios.get("http://localhost/api/sharks/heatmap/");
  const layer = new HeatmapLayer({
    parent: deckContainer.value,
    data: 'http://localhost/api/sharks/heatmap/',
    aggregation: 'SUM',
    getPosition: d => [d.lat, d.lon],
    getWeight: d => d.weight,
    radiusPixels: 25
  });
  layers.value = [oceanBasemap, layer];
}

loadHeatmap();

watch(layers, (val) => {
  console.log(val);
  deck.value.setProps({ layers: val })
})

deck.value = new Deck({
  parent: deckContainer.value,
  initialViewState: INITIAL_VIEW_STATE,
  controller: true,
  views: [new MapView({repeat: true})],
  layers: layers.value,
  onClick: info => {
    popUp.value.showMenu();
    coords.value = {
      x: info.x,
      y: info.y,
    };
  },
});

onBeforeUnmount(() => {
  deck.value?.finalize();
  deck.value = null;
});
</script>

<template>
  <div>
    <div ref="deckContainer"></div>
<!--    <PopUp ref="popUp" :deck="deck" :coordinates="coords"/>-->
  </div>
</template>
